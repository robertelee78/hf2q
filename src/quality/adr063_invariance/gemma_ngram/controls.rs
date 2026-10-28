//! Hosted source controls; synthetic inputs are not actual speculative proof.
use super::*;
#[test]
fn fixture_changes_only_predetermined_proposals_never_acceptance(){
    let canonical=(0..144).map(|n|1000+n).collect::<Vec<u32>>();
    let d=fixture(&canonical,262144).unwrap();
    assert_eq!(&d[..2],&canonical[129..131]);assert_ne!(d[2],canonical[131]);
    assert_eq!(canonical[131],1131);assert!(![1,106].contains(&d[2]));
    assert!(fixture(&canonical[..143],262144).is_err());
    let mut bad=canonical.clone();bad[130]=262144;assert!(fixture(&bad,262144).is_err());
}
#[test]
fn zero_acceptance_does_not_become_accepted_transition_coverage(){
    use crate::inference::spec_decode::verifier::accept_prefix_argmax;
    let (n,fallback)=accept_prefix_argmax(&[10,20,30],&[11,20,30,40]);
    assert_eq!(n,0);assert_eq!(fallback,11);assert_eq!(coverage(n,3-n),(false,true));
    let (n,fallback)=accept_prefix_argmax(&[10,20,30],&[10,20,31,40]);
    assert_eq!(n,2);assert_eq!(fallback,31);assert_eq!(coverage(n,3-n),(true,true));
}
#[test]
fn actual_commit_coordinate_gate_catches_untrimmed_and_wrong_fallback_cursor(){
    for accepted in 0..=3 {let rejected=3-accepted;require_commit(129+accepted,130+accepted,rejected).unwrap();}
    assert!(require_commit(132,130,3).is_err());assert!(require_commit(128,130,3).is_err());
    assert!(require_commit(129,130,4).is_err());assert!(require_commit(0,0,0).is_err());
}
#[test]
fn actual_phase_gate_refuses_decision_or_commit_without_checked_verifier(){
    assert!(transition(Phase::Verified,Phase::Verified).is_ok());
    for stage in [Phase::Ready,Phase::Initial,Phase::Proposed,Phase::Verifying,Phase::Decided,Phase::Committed] {
        assert!(transition(stage,Phase::Verified).is_err());
    }
    assert!(super::ACTIVE.with(|a|a.borrow().is_none()));
    assert_eq!(proposal(&[],&[],3).unwrap(),None);
}
#[test]
fn finite_active_role_formula_is_below_whole_case_payload_limit(){
    let upper=planned_payload_upper();
    assert_eq!(upper,1557187584);assert!(upper<PAYLOAD);assert_eq!(METADATA,134217728);
}

#[test]
fn actual_selected_profile_guard_refuses_other_cache_policies_before_load(){
    require_selected_profile(true,false,8,8,false).unwrap();
    for (hybrid,dense,alloc,effective,full) in [(false,false,8,8,false),(true,true,8,8,false),
        (true,false,0,8,false),(true,false,4,4,false),(true,false,5,5,false),
        (true,false,6,6,false),(true,false,8,4,false),(true,false,8,8,true)] {
        assert!(require_selected_profile(hybrid,dense,alloc,effective,full).is_err());
    }
    for raw in [Some("1"),Some("true"),Some("on")] {
        assert!(full_f16_enabled(raw));
        assert!(require_selected_profile(true,false,8,8,full_f16_enabled(raw)).is_err());
    }
    for raw in [None,Some("0"),Some("false"),Some("off")] {assert!(!full_f16_enabled(raw));}
}
