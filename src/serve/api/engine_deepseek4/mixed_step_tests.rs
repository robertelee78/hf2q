//! ADR-063 S1: model-free session-ledger half of a mixed step.
//!
//! The cache-transaction half (`deepseek4::mixed_step_tests`) proves the
//! publishers isolate disjoint lanes, and that a lane drafted into both
//! transactions of one step poisons an innocent decoder. These fixtures pin
//! the session guards that keep the cooperative cohort and the joint decode
//! set disjoint, and prove that a member's ledger publishes exactly,
//! publishes nothing, or rolls back without touching a decoding nonmember.

use mlx_native::{DType, MlxDevice};

use crate::inference::models::deepseek4::cache::{Deepseek4Cache, Deepseek4CacheSnapshot};
use crate::inference::models::deepseek4::mixed_step_tests::host_cache_plan;

use super::Deepseek4Session;

const LANES: usize = 5;
const MEMBERS: [usize; 3] = [0, 2, 3];
const NONMEMBERS: [usize; 2] = [1, 4];
const START: [usize; LANES] = [148, 125, 148, 148, 254];
const COHORT_ROWS: usize = 128;
const VOCAB: usize = 8;

fn prompt(lane: usize, rows: usize) -> Vec<u32> {
    (0..rows)
        .map(|row| ((row * 97 + lane * 193 + 11) % 120_000) as u32)
        .collect()
}

fn session(device: &MlxDevice, tokens: &[u32], decoding: bool) -> Deepseek4Session {
    let mut cache = Deepseek4Cache::allocate(&host_cache_plan(), device.clone()).unwrap();
    cache.commit_prefill(0, tokens.len()).unwrap();
    let live_logits = decoding.then(|| {
        device
            .alloc_buffer(VOCAB * DType::F32.size_of(), DType::F32, vec![1, VOCAB])
            .unwrap()
    });
    Deepseek4Session {
        cache,
        committed_tokens: tokens.to_vec(),
        live_logits,
        turn_anchor: None,
        turn_anchor_tokens: Vec::new(),
        pending_turn_anchor: None,
        pending_turn_anchor_tokens: Vec::new(),
        keep_turn_anchor_on_success: false,
        request_anchor_transaction_active: false,
        disk_persistor: None,
    }
}

/// Everything a decoding nonmember exposes to the scheduler and sampler.
#[derive(Debug, PartialEq)]
struct LedgerView {
    tokens: Vec<u32>,
    cache_position: usize,
    poisoned: bool,
    logits: Option<usize>,
    anchor: Option<usize>,
    anchor_tokens: Vec<u32>,
}

fn view(session: &Deepseek4Session) -> LedgerView {
    LedgerView {
        tokens: session.committed_tokens.clone(),
        cache_position: session.cache.position(),
        poisoned: session.cache.is_poisoned(),
        logits: session
            .live_logits
            .as_ref()
            .map(|logits| logits.contents_ptr() as usize),
        anchor: session
            .turn_anchor
            .as_ref()
            .map(Deepseek4CacheSnapshot::position),
        anchor_tokens: session.turn_anchor_tokens.clone(),
    }
}

#[test]
fn decoding_nonmember_cannot_be_drafted_into_a_cooperative_cohort() {
    let _gpu = crate::inference::hf2q_gpu_test_lock();
    let device = MlxDevice::new().unwrap();

    let mut decoder = session(&device, &prompt(1, START[1]), true);
    let before = view(&decoder);
    assert!(decoder.joint_decode_ready());
    assert!(decoder
        .prepare_cooperative_prefill(START[1], COHORT_ROWS)
        .is_err());
    assert_eq!(view(&decoder), before);

    let mut member = session(&device, &prompt(0, START[0]), false);
    member
        .prepare_cooperative_prefill(START[0], COHORT_ROWS)
        .unwrap();
    // From planning until the serial tail installs logits, a member is never
    // eligible for the joint decode set of the same or a later step.
    assert!(!member.joint_decode_ready());
    member.cache.commit_prefill(START[0], COHORT_ROWS).unwrap();
    assert!(!member.joint_decode_ready());
    member
        .validate_cooperative_prefill_result(START[0], COHORT_ROWS)
        .unwrap();
    member.publish_cooperative_tokens(&prompt(0, START[0] + COHORT_ROWS)[START[0]..]);
    assert!(!member.joint_decode_ready());
    // A stale cooperative plan cannot be replayed onto the published member.
    assert!(member
        .prepare_cooperative_prefill(START[0], COHORT_ROWS)
        .is_err());
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Outcome {
    /// The model settler published every member cursor.
    Committed,
    /// Late failure or poison: the settler published no member cursor.
    Rejected,
    /// The cohort committed, then two members' clients cancelled.
    Cancelled,
}

#[test]
fn cohort_ledger_commits_exactly_publishes_nothing_or_rolls_back_alone() {
    let _gpu = crate::inference::hf2q_gpu_test_lock();
    let device = MlxDevice::new().unwrap();
    let anchored = MEMBERS[1];
    let unanchored = MEMBERS[2];
    for outcome in [Outcome::Committed, Outcome::Rejected, Outcome::Cancelled] {
        let mut sessions = (0..LANES)
            .map(|lane| {
                session(
                    &device,
                    &prompt(lane, START[lane]),
                    NONMEMBERS.contains(&lane),
                )
            })
            .collect::<Vec<_>>();
        sessions[anchored].turn_anchor = Some(sessions[anchored].cache.snapshot().unwrap());
        sessions[anchored].turn_anchor_tokens = prompt(anchored, START[anchored]);
        let decoders = NONMEMBERS.map(|lane| view(&sessions[lane]));

        for lane in MEMBERS {
            sessions[lane]
                .prepare_cooperative_prefill(START[lane], COHORT_ROWS)
                .unwrap();
        }
        // Stand-in for the model settler, which publishes every member
        // cursor or none (cursor/poison semantics are proven model-free in
        // `deepseek4::mixed_step_tests`).
        if outcome != Outcome::Rejected {
            for lane in MEMBERS {
                sessions[lane]
                    .cache
                    .commit_prefill(START[lane], COHORT_ROWS)
                    .unwrap();
            }
        }
        let validated = MEMBERS.map(|lane| {
            sessions[lane]
                .validate_cooperative_prefill_result(START[lane], COHORT_ROWS)
                .is_ok()
        });
        assert_eq!(validated, [outcome != Outcome::Rejected; 3], "{outcome:?}");

        for lane in MEMBERS {
            if outcome == Outcome::Rejected {
                assert_eq!(sessions[lane].committed_tokens, prompt(lane, START[lane]));
                assert_eq!(sessions[lane].cache.position(), START[lane]);
            } else {
                let tokens = prompt(lane, START[lane] + COHORT_ROWS);
                sessions[lane].publish_cooperative_tokens(&tokens[START[lane]..]);
                assert_eq!(sessions[lane].committed_tokens, tokens, "{outcome:?}");
            }
            assert!(sessions[lane].live_logits.is_none());
        }

        if outcome == Outcome::Cancelled {
            sessions[anchored].recover_after_cancellation().unwrap();
            sessions[unanchored].recover_after_cancellation().unwrap();
            assert_eq!(sessions[anchored].cache.position(), START[anchored]);
            assert_eq!(
                sessions[anchored].committed_tokens,
                prompt(anchored, START[anchored])
            );
            assert!(!sessions[anchored].cache.is_poisoned());
            assert_eq!(sessions[unanchored].cache.position(), 0);
            assert!(sessions[unanchored].committed_tokens.is_empty());
            let kept = MEMBERS[0];
            assert_eq!(
                sessions[kept].committed_tokens,
                prompt(kept, START[kept] + COHORT_ROWS)
            );
            assert_eq!(sessions[kept].cache.position(), START[kept] + COHORT_ROWS);
        }

        for (lane, before) in NONMEMBERS.into_iter().zip(&decoders) {
            assert_eq!(&view(&sessions[lane]), before, "{outcome:?} decoder {lane}");
            assert!(sessions[lane].joint_decode_ready());
        }
    }
}
