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
