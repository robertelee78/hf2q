use serde_json::json;

use super::macho::{verify_bytes, MachOError};
use crate::distribution::schema::ReleaseManifestV1;

mod fixture;
mod hostile_cases;

use fixture::*;

const LC_SYMTAB: u32 = 0x02;
const LC_DYSYMTAB: u32 = 0x0b;
const LC_LOAD_DYLIB: u32 = 0x0c;
const LC_LOAD_DYLINKER: u32 = 0x0e;
const LC_LOADFVMLIB: u32 = 0x06;
const LC_PREBOUND_DYLIB: u32 = 0x10;
const LC_UNIXTHREAD: u32 = 0x05;
const LC_LOAD_WEAK_DYLIB: u32 = 0x8000_0018;
const LC_SEGMENT_64: u32 = 0x19;
const LC_UUID: u32 = 0x1b;
const LC_RPATH: u32 = 0x8000_001c;
const LC_CODE_SIGNATURE: u32 = 0x1d;
const LC_REEXPORT_DYLIB: u32 = 0x8000_001f;
const LC_LAZY_LOAD_DYLIB: u32 = 0x20;
const LC_ENCRYPTION_INFO_64: u32 = 0x2c;
const LC_DYLD_INFO_ONLY: u32 = 0x8000_0022;
const LC_LOAD_UPWARD_DYLIB: u32 = 0x8000_0023;
const LC_VERSION_MIN_IPHONEOS: u32 = 0x25;
const LC_FUNCTION_STARTS: u32 = 0x26;
const LC_MAIN: u32 = 0x8000_0028;
const LC_DATA_IN_CODE: u32 = 0x29;
const LC_VERSION_MIN_TVOS: u32 = 0x2f;
const LC_VERSION_MIN_WATCHOS: u32 = 0x30;
const LC_BUILD_VERSION: u32 = 0x32;
const MH_NOUNDEFS: u32 = 0x0000_0001;
const MH_DYLDLINK: u32 = 0x0000_0004;
const MH_TWOLEVEL: u32 = 0x0000_0080;
const MH_PIE: u32 = 0x0020_0000;

#[derive(Clone)]
struct FixtureOptions {
    cpu_subtype: u32,
    file_type: u32,
    header_flags: u32,
    minimum_macos: u32,
    sdk_macos: u32,
    segment_max_protection: u32,
    segment_initial_protection: u32,
    dylib: &'static str,
    dylinker: &'static str,
    extra_commands: Vec<Vec<u8>>,
    omitted_metadata_command: Option<u32>,
    include_build_version: bool,
    include_main: bool,
    main_stack_size: u64,
    include_code_signature: bool,
}

impl Default for FixtureOptions {
    fn default() -> Self {
        Self {
            cpu_subtype: 0,
            file_type: 2,
            header_flags: MH_NOUNDEFS | MH_DYLDLINK | MH_TWOLEVEL | MH_PIE,
            minimum_macos: 14 << 16,
            sdk_macos: 26 << 16,
            segment_max_protection: 5,
            segment_initial_protection: 5,
            dylib: "/usr/lib/libSystem.B.dylib",
            dylinker: "/usr/lib/dyld",
            extra_commands: Vec::new(),
            omitted_metadata_command: None,
            include_build_version: true,
            include_main: true,
            main_stack_size: 0,
            include_code_signature: true,
        }
    }
}

#[test]
fn exact_thin_arm64_system_only_profile_is_accepted() {
    let bytes = fixture(FixtureOptions::default());
    let manifest = manifest(bytes.len() as u64, "14.0", false);
    let verified = verify_bytes(&bytes, &manifest).expect("valid thin arm64 Mach-O");
    assert_eq!(verified.code_signature_range().end, bytes.len() as u64);
}
#[test]
fn architecture_header_and_executable_memory_policy_fail_closed() {
    let cases = [
        FixtureOptions {
            cpu_subtype: 2,
            ..FixtureOptions::default()
        },
        FixtureOptions {
            file_type: 6,
            ..FixtureOptions::default()
        },
        FixtureOptions {
            header_flags: MH_NOUNDEFS | MH_DYLDLINK | MH_TWOLEVEL | MH_PIE | 0x0002_0000,
            ..FixtureOptions::default()
        },
        FixtureOptions {
            header_flags: 0,
            ..FixtureOptions::default()
        },
        FixtureOptions {
            segment_max_protection: 7,
            ..FixtureOptions::default()
        },
        FixtureOptions {
            segment_initial_protection: 7,
            ..FixtureOptions::default()
        },
    ];
    for options in cases {
        let bytes = fixture(options);
        assert_invalid(&bytes, &manifest(bytes.len() as u64, "14.0", false));
    }

    let required_flags = MH_NOUNDEFS | MH_DYLDLINK | MH_TWOLEVEL | MH_PIE;
    for missing in [MH_NOUNDEFS, MH_DYLDLINK, MH_TWOLEVEL, MH_PIE] {
        let bytes = fixture(FixtureOptions {
            header_flags: required_flags & !missing,
            ..FixtureOptions::default()
        });
        assert_invalid(&bytes, &manifest(bytes.len() as u64, "14.0", false));
    }

    let force_flat = fixture(FixtureOptions {
        header_flags: required_flags | 0x0000_0100,
        ..FixtureOptions::default()
    });
    assert_invalid(
        &force_flat,
        &manifest(force_flat.len() as u64, "14.0", false),
    );

    let mut wrong_magic = fixture(FixtureOptions::default());
    wrong_magic[0..4].copy_from_slice(&0xcafebabe_u32.to_le_bytes());
    assert_invalid(
        &wrong_magic,
        &manifest(wrong_magic.len() as u64, "14.0", false),
    );
}

#[test]
fn deployment_target_and_manifest_dependency_claims_are_exact() {
    let bytes = fixture(FixtureOptions::default());
    assert_invalid(&bytes, &manifest(bytes.len() as u64, "14.1", false));
    assert_invalid(&bytes, &manifest(bytes.len() as u64, "14.0", true));

    let patch = fixture(FixtureOptions {
        minimum_macos: (14 << 16) | 1,
        ..FixtureOptions::default()
    });
    verify_bytes(&patch, &manifest(patch.len() as u64, "14.0.1", false))
        .expect("canonical nonzero patch deployment target");

    let sdk_below_minimum = fixture(FixtureOptions {
        sdk_macos: 13 << 16,
        ..FixtureOptions::default()
    });
    assert_invalid(
        &sdk_below_minimum,
        &manifest(sdk_below_minimum.len() as u64, "14.0", false),
    );

    let unrepresentable = fixture(FixtureOptions::default());
    assert_invalid(
        &unrepresentable,
        &manifest(unrepresentable.len() as u64, "14.256", false),
    );
}

#[test]
fn only_public_system_dependencies_and_the_canonical_dylinker_are_accepted() {
    for dylib in [
        "@rpath/libmlx.dylib",
        "@loader_path/libmlx.dylib",
        "@executable_path/libmlx.dylib",
        "/opt/homebrew/lib/libmlx.dylib",
        "/nix/store/example/libmlx.dylib",
        "/System/Library/PrivateFrameworks/Foo.framework/Foo",
        "/usr/lib/../local/libbad.dylib",
    ] {
        let bytes = fixture(FixtureOptions {
            dylib,
            ..FixtureOptions::default()
        });
        assert_invalid(&bytes, &manifest(bytes.len() as u64, "14.0", false));
    }

    let framework = fixture(FixtureOptions {
        dylib: "/System/Library/Frameworks/Metal.framework/Versions/A/Metal",
        ..FixtureOptions::default()
    });
    verify_bytes(&framework, &manifest(framework.len() as u64, "14.0", false))
        .expect("public Apple framework");

    let wrong_dylinker = fixture(FixtureOptions {
        dylinker: "/usr/local/lib/dyld",
        ..FixtureOptions::default()
    });
    assert_invalid(
        &wrong_dylinker,
        &manifest(wrong_dylinker.len() as u64, "14.0", false),
    );

    for kind in [
        LC_LOAD_WEAK_DYLIB,
        LC_REEXPORT_DYLIB,
        LC_LAZY_LOAD_DYLIB,
        LC_LOAD_UPWARD_DYLIB,
    ] {
        let variant = fixture(FixtureOptions {
            extra_commands: vec![dylib_command_kind(kind, "/usr/lib/libSystem.B.dylib")],
            ..FixtureOptions::default()
        });
        assert_invalid(&variant, &manifest(variant.len() as u64, "14.0", false));
    }
}
#[test]
fn rpath_encryption_and_legacy_or_duplicate_metadata_are_rejected() {
    let rpath = fixture(FixtureOptions {
        extra_commands: vec![string_command(LC_RPATH, "/tmp")],
        ..FixtureOptions::default()
    });
    assert_invalid(&rpath, &manifest(rpath.len() as u64, "14.0", false));

    let encryption = fixture(FixtureOptions {
        extra_commands: vec![fixed_command(LC_ENCRYPTION_INFO_64, &[0; 16])],
        ..FixtureOptions::default()
    });
    assert_invalid(
        &encryption,
        &manifest(encryption.len() as u64, "14.0", false),
    );

    let missing_build = fixture(FixtureOptions {
        include_build_version: false,
        ..FixtureOptions::default()
    });
    assert_invalid(
        &missing_build,
        &manifest(missing_build.len() as u64, "14.0", false),
    );

    let duplicate_build = fixture(FixtureOptions {
        extra_commands: vec![build_version(14 << 16, 26 << 16)],
        ..FixtureOptions::default()
    });
    assert_invalid(
        &duplicate_build,
        &manifest(duplicate_build.len() as u64, "14.0", false),
    );

    for kind in [
        LC_UNIXTHREAD,
        LC_LOADFVMLIB,
        LC_PREBOUND_DYLIB,
        LC_VERSION_MIN_IPHONEOS,
        LC_VERSION_MIN_TVOS,
        LC_VERSION_MIN_WATCHOS,
    ] {
        let legacy = fixture(FixtureOptions {
            extra_commands: vec![fixed_command(kind, &[])],
            ..FixtureOptions::default()
        });
        assert_invalid(&legacy, &manifest(legacy.len() as u64, "14.0", false));
    }
}
