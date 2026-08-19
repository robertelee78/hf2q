use super::codesign::{
    inspect_apple_binary_for_test, validate_signing_info, verify_path, CodeSigningError,
    SigningInfoView, SigningPolicy,
};
use crate::distribution::schema::ReleaseManifestV1;

const TEAM_ID: &str = "A1B2C3D4E5";
const IDENTIFIER: &str = "us.hf2q.cli";

fn manifest() -> ReleaseManifestV1 {
    ReleaseManifestV1::parse_and_validate(include_bytes!(
        "../schema/testdata/release_manifest_v1.json"
    ))
    .expect("canonical release manifest fixture")
}

fn valid_view() -> SigningInfoView {
    SigningInfoView {
        identifier_matches: true,
        team_matches: true,
        flags: Some(0x1_0000),
        timestamp_is_date: true,
        raw_entitlements_absent: true,
        dictionary_entitlements_absent: true,
        certificate_chain_length: Some(3),
        leaf_common_name_matches: true,
    }
}

#[test]
fn compiled_policy_precedes_native_code_lookup() {
    let manifest = manifest();
    let matching = SigningPolicy::for_test(TEAM_ID, IDENTIFIER).expect("matching test policy");
    matching
        .require_manifest(&manifest)
        .expect("manifest repeats the exact policy");

    let wrong_team = SigningPolicy::for_test("Z9Y8X7W6V5", IDENTIFIER).expect("test policy");
    assert_eq!(
        wrong_team.require_manifest(&manifest),
        Err(CodeSigningError::Policy)
    );
    let wrong_identifier = SigningPolicy::for_test(TEAM_ID, "us.hf2q.other").expect("test policy");
    assert_eq!(
        wrong_identifier.require_manifest(&manifest),
        Err(CodeSigningError::Policy)
    );
}

#[test]
fn code_requirement_is_exact_and_uses_no_manifest_text() {
    let policy = SigningPolicy::for_test(TEAM_ID, IDENTIFIER).expect("test policy");
    assert_eq!(
        policy.requirement(),
        "anchor apple generic and anchor trusted and identifier \"us.hf2q.cli\" and certificate 1[field.1.2.840.113635.100.6.2.6] exists and certificate leaf[field.1.2.840.113635.100.6.1.13] exists and certificate leaf[subject.OU] = \"A1B2C3D4E5\""
    );
}

#[test]
fn test_policy_rejects_requirement_injection_and_noncanonical_team_ids() {
    for team_id in ["SHORT", "a1B2C3D4E5", "A1B2C3D4E!"] {
        assert!(matches!(
            SigningPolicy::for_test(team_id, IDENTIFIER),
            Err(CodeSigningError::Policy)
        ));
    }
    for identifier in [
        "",
        "us.hf2q.\" or anchor apple",
        "us/hf2q/cli",
        "us.hf2q.cli\\escape",
    ] {
        assert!(matches!(
            SigningPolicy::for_test(TEAM_ID, identifier),
            Err(CodeSigningError::Policy)
        ));
    }
}

#[test]
fn signing_information_is_a_closed_fail_closed_profile() {
    validate_signing_info(&valid_view()).expect("complete valid signing information");

    let mut invalid = Vec::new();
    let mut case = valid_view();
    case.identifier_matches = false;
    invalid.push(case);
    let mut case = valid_view();
    case.team_matches = false;
    invalid.push(case);
    let mut case = valid_view();
    case.flags = None;
    invalid.push(case);
    let mut case = valid_view();
    case.flags = Some(0);
    invalid.push(case);
    let mut case = valid_view();
    case.flags = Some(0x1_0000 | 0x0002);
    invalid.push(case);
    let mut case = valid_view();
    case.flags = Some(0x1_0000 | 0x2_0000);
    invalid.push(case);
    let mut case = valid_view();
    case.timestamp_is_date = false;
    invalid.push(case);
    let mut case = valid_view();
    case.raw_entitlements_absent = false;
    invalid.push(case);
    let mut case = valid_view();
    case.dictionary_entitlements_absent = false;
    invalid.push(case);
    let mut case = valid_view();
    case.certificate_chain_length = None;
    invalid.push(case);
    let mut case = valid_view();
    case.certificate_chain_length = Some(0);
    invalid.push(case);
    let mut case = valid_view();
    case.certificate_chain_length = Some(9);
    invalid.push(case);
    let mut case = valid_view();
    case.leaf_common_name_matches = false;
    invalid.push(case);

    for case in invalid {
        assert_eq!(
            validate_signing_info(&case),
            Err(CodeSigningError::InvalidSignature)
        );
    }
}

#[test]
fn manifest_policy_failure_happens_before_path_lookup() {
    let manifest = manifest();
    let policy = SigningPolicy::for_test("Z9Y8X7W6V5", IDENTIFIER).expect("test policy");
    assert!(matches!(
        verify_path(
            std::path::Path::new("/path/that/must/not/be-opened"),
            &manifest,
            &policy,
            crate::distribution::install_state::ExecutableReleaseBinding::for_test(),
        ),
        Err(CodeSigningError::Policy)
    ));
}

#[test]
fn current_test_binary_is_not_accepted_as_a_developer_id_release() {
    let path = std::env::current_exe().expect("current test executable path");
    let manifest = manifest();
    let policy = SigningPolicy::for_test(TEAM_ID, IDENTIFIER).expect("test policy");
    assert!(matches!(
        verify_path(
            &path,
            &manifest,
            &policy,
            crate::distribution::install_state::ExecutableReleaseBinding::for_test(),
        ),
        Err(CodeSigningError::InvalidSignature)
    ));
}

#[test]
fn system_apple_binary_exercises_the_typed_signing_information_bridge() {
    let view = inspect_apple_binary_for_test(std::path::Path::new("/bin/ls"))
        .expect("Apple-signed system binary");
    assert!(!view.identifier_matches);
    assert!(!view.team_matches);
    assert!(view.flags.is_some());
    assert!(matches!(view.certificate_chain_length, Some(1..=8)));
    assert!(!view.leaf_common_name_matches);
    assert_eq!(
        validate_signing_info(&view),
        Err(CodeSigningError::InvalidSignature)
    );
}

#[test]
#[ignore = "requires an exact protected Developer ID Application release fixture"]
fn protected_developer_id_release_fixture_is_accepted() {
    let path = std::env::var_os("HF2Q_DEVELOPER_ID_TEST_BINARY")
        .expect("HF2Q_DEVELOPER_ID_TEST_BINARY is required");
    let team_id =
        std::env::var("HF2Q_DEVELOPER_ID_TEST_TEAM_ID").expect("test Team ID is required");
    let common_name = std::env::var("HF2Q_DEVELOPER_ID_TEST_COMMON_NAME")
        .expect("test certificate common name is required");
    let mut raw: serde_json::Value = serde_json::from_slice(include_bytes!(
        "../schema/testdata/release_manifest_v1.json"
    ))
    .expect("manifest fixture JSON");
    raw["code_signing"]["team_id"] = serde_json::Value::String(team_id.clone());
    raw["code_signing"]["certificate_common_name"] = serde_json::Value::String(common_name);
    let manifest = ReleaseManifestV1::parse_and_validate(
        &serde_json::to_vec(&raw).expect("test manifest encoding"),
    )
    .expect("test manifest");
    let policy = SigningPolicy::for_test(&team_id, IDENTIFIER).expect("test signing policy");
    verify_path(
        std::path::Path::new(&path),
        &manifest,
        &policy,
        crate::distribution::install_state::ExecutableReleaseBinding::for_test(),
    )
    .expect("protected Developer ID fixture");
}
