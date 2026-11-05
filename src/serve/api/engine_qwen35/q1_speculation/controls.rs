//! Hosted controls exercise the actual hook guards without a device or model.
use super::*;
fn active(mode: Mode) -> Active {
    Active::new(
        mode,
        qwen_cache::selected_header_fixture(true),
        (0..4).map(|n| vec![n; 128]).collect(),
        None,
        vec![],
        None,
        None,
    )
}
#[test]
fn q1_t5_actual_transition_gate_rejects_fabricated_success_and_wrong_rollback() {
    let plan = Qwen35VerifiedBlockPlan {
        output: VecDeque::from([7, 9]),
        terminal_after_pending: false,
        matched_drafts: 1,
        rejected_drafts: 1,
        valid_input_tokens: 2,
        carry_hidden_row: 1,
    };
    assert!(accepted_rejected_transition(
        Mode::ForcedProposal,
        129,
        131,
        &plan,
        2
    ));
    for (mode, pos, cursor, n) in [
        (Mode::OrdinaryBaseline, 129, 131, 2),
        (Mode::ForcedProposal, 128, 131, 2),
        (Mode::ForcedProposal, 129, 133, 2),
        (Mode::ForcedProposal, 129, 131, 4),
    ] {
        assert!(!accepted_rejected_transition(mode, pos, cursor, &plan, n));
    }
    let mut wrong = plan;
    wrong.carry_hidden_row = 0;
    assert!(!accepted_rejected_transition(
        Mode::ForcedProposal,
        129,
        131,
        &wrong,
        2
    ));
}
#[test]
fn q1_t5_completion_rejects_omitted_rows_and_unconsumed_raw_tail() {
    let mut a = active(Mode::OrdinaryBaseline);
    for lane in 0..4 {
        a.selected[lane] = vec![
            Carry {
                token: 1,
                terminal: false
            };
            17
        ];
        a.ordinary_rows[lane] = (128..144).collect();
    }
    assert!(runner::validate_complete(&a).is_ok());
    a.ordinary_rows[2].remove(5);
    assert!(runner::validate_complete(&a).is_err());
    a.ordinary_rows[2] = (128..144).collect();
    a.pending_logits = Some(vec![1.0]);
    assert!(runner::validate_complete(&a).is_err());
    a.pending_logits = None;
    a.selected[3].pop();
    assert!(runner::validate_complete(&a).is_err());
}
#[test]
fn q1_t5_canonical_lineage_and_final_carry_are_checked_not_extra_input() {
    let mut a = active(Mode::OrdinaryBaseline);
    let vocab = a.cfg.vocab_size;
    a.frozen = Some(Frozen {
        workload: Workload {
            family: Family::Qwen38,
            vocab_size: vocab,
            lanes: (0..4)
                .map(|lane| {
                    let mut p = vec![lane; 128];
                    p.extend(10..26);
                    p
                })
                .collect(),
        },
        final_carry: [Carry {
            token: 26,
            terminal: true,
        }; 4],
        forced_bad_ids: [0; 4],
    });
    assert!(selected(&mut a, 127, 0, 11, false).is_err());
    assert!(a.selected[0].is_empty());
    for token in 10..26 {
        selected(&mut a, 128, 0, token, false).unwrap();
    }
    assert!(selected(&mut a, 143, 0, 26, false).is_err());
    selected(&mut a, 143, 0, 26, true).unwrap();
    assert_eq!(a.selected[0].len(), 17);
    assert!(selected(&mut a, 144, 0, 27, false).is_err());
}
#[test]
fn q1_t5_hook_scope_is_panic_safe_and_inactive_proposals_unchanged() {
    assert_eq!(after_drafts(0, 0, 0, vec![4, 5, 6]).unwrap(), [4, 5, 6]);
    let result = std::panic::catch_unwind(|| {
        let _scope = install(active(Mode::Preparation)).unwrap();
        assert!(install(active(Mode::Preparation)).is_err());
        panic!("injected after install");
    });
    assert!(result.is_err());
    assert!(ACTIVE.with(|s| s.borrow().is_none()));
    let old = install(active(Mode::Preparation)).unwrap();
    let _finished = finish().unwrap();
    let current = install(active(Mode::Preparation)).unwrap();
    drop(old);
    assert!(ACTIVE.with(|s| s.borrow().is_some()));
    drop(current);
    assert!(ACTIVE.with(|s| s.borrow().is_none()));
}
#[test]
fn q1_t5_first_fork_inputs_bind_reviewed_frozen_preparation() {
    use crate::adr063_invariance::store::hash;
    let (frozen, digest) = (runner::Q38_RETAINED_FROZEN_SHA256, runner::Q38_RETAINED_FULL144_DIGEST);
    let preparation = |frozen: &str, digest: &str, comparable: bool| serde_json::to_vec(&serde_json::json!({
        "frozen_file_sha256":frozen,"full144_digest_computed_after_generation":digest,
        "comparable_acquisition":comparable,"requires_independent_review":true})).unwrap();
    let review = |verdict: &str, frozen: &str, prep: &[u8]| serde_json::to_vec(&serde_json::json!({
        "schema":runner::FROZEN_REVIEW_SCHEMA,"verdict":verdict,"frozen_file_sha256":frozen,
        "preparation_sha256":hash(prep),"reviewer":"fixture-reviewer","notes":"hosted fixture"})).unwrap();
    let ok = |first_fork: bool, p: &[u8], r: &[u8]|
        runner::check_first_fork_inputs(first_fork, Some((p, r)), frozen, digest).is_ok();
    // Control: the retained Q38 literals are accepted through the rule.
    let p = preparation(frozen, digest, false);
    let r = review("Accepted", frozen, &p);
    assert!(ok(true, &p, &r));
    assert!(runner::check_first_fork_inputs(false, None, frozen, digest).is_ok());
    // Missing with first_fork; present without it.
    assert!(runner::check_first_fork_inputs(true, None, frozen, digest).is_err());
    assert!(!ok(false, &p, &r));
    // Review verdict other than Accepted.
    assert!(!ok(true, &p, &review("Rejected", frozen, &p)));
    // Frozen-sha mismatch: in the review, in the preparation, or in the input.
    let other = "1".repeat(64);
    assert!(!ok(true, &p, &review("Accepted", &other, &p)));
    let q = preparation(&other, digest, false);
    assert!(!ok(true, &q, &review("Accepted", frozen, &q)));
    assert!(runner::check_first_fork_inputs(true, Some((p.as_slice(), r.as_slice())), &other, digest).is_err());
    // Preparation digest, comparability, review/preparation binding, closed review.
    let q = preparation(frozen, &other, false);
    assert!(!ok(true, &q, &review("Accepted", frozen, &q)));
    let q = preparation(frozen, digest, true);
    assert!(!ok(true, &q, &review("Accepted", frozen, &q)));
    let mut q = p.clone();
    q.push(b'\n');
    assert!(!ok(true, &q, &r), "review binds the exact preparation bytes");
    let mut open: serde_json::Value = serde_json::from_slice(&r).unwrap();
    open["extra"] = serde_json::json!(true);
    assert!(!ok(true, &p, &serde_json::to_vec(&open).unwrap()));
    let mut schema: serde_json::Value = serde_json::from_slice(&r).unwrap();
    schema["schema"] = serde_json::json!("hf2q.q1.other.v1");
    assert!(!ok(true, &p, &serde_json::to_vec(&schema).unwrap()));
}
