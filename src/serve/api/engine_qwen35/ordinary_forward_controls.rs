//! Hosted controls of the actual pre-catchup validation; no model/GPU fixture.
use super::*;
fn fingerprint() -> DecodeFingerprint {
    DecodeFingerprint { slot: SlotId(2), step: 1, generated_len: 1, last_token: 1234, position_base: 128 }
}
fn validate(prior: (u32,u32), now: (u32,u32)) -> Result<()> {
    validate_boundary(fingerprint(), fingerprint(), prior, now,
        (248320,5120,true), (248320,5120), true)
}
#[test]
fn actual_post_target_boundary_accepts_only_target_advanced_not_mtp() {
    validate((128,128),(129,128)).unwrap();
    assert!(validate((128,128),(128,128)).is_err()); // missing target
    assert!(validate((128,128),(130,128)).is_err()); // double target
    assert!(validate((128,128),(129,129)).is_err()); // catchup already ran
}
#[test]
fn advanced_target_cannot_be_reread_as_old_prior() {
    validate((128,128),(129,128)).unwrap();
    assert!(validate((129,128),(129,128)).is_err());
    assert!(validate((129,129),(129,128)).is_err());
}
#[test]
fn prepared_slot_step_input_and_position_refuse_cross_state_completion() {
    for current in [DecodeFingerprint { slot: SlotId(1), ..fingerprint() },
        DecodeFingerprint { step: 2, ..fingerprint() },
        DecodeFingerprint { generated_len: 2, ..fingerprint() },
        DecodeFingerprint { last_token: 4321, ..fingerprint() },
        DecodeFingerprint { position_base: 129, ..fingerprint() }] {
        assert!(validate_boundary(fingerprint(),current,(128,128),(129,128),
            (248320,5120,true),(248320,5120),true).is_err());
    }
}
#[test]
fn exact_full_logits_and_one_f32_normalized_row_are_required() {
    for output in [(248319,5120,true),(248320,10240,true),(248320,5120,false),(0,5120,true)] {
        assert!(validate_boundary(fingerprint(),fingerprint(),(128,128),(129,128),
            output,(248320,5120),true).is_err());
    }
    assert!(validate_boundary(fingerprint(),fingerprint(),(128,128),(129,128),
        (248320,5120,true),(248320,5120),false).is_err());
}
#[test]
fn cursor_overflow_and_incoherent_prior_fail_before_catchup() {
    assert!(validate((u32::MAX,u32::MAX),(0,u32::MAX)).is_err());
    assert!(validate((128,127),(129,127)).is_err());
}
