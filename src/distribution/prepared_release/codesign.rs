use std::ffi::c_void;
use std::path::Path;
use std::ptr;
use std::str::FromStr;

use core_foundation::array::{CFArray, CFArrayGetValueAtIndex};
use core_foundation::base::{CFEqual, CFGetTypeID, CFTypeID, CFTypeRef, TCFType};
use core_foundation::date::CFDateGetTypeID;
use core_foundation::dictionary::{CFDictionary, CFDictionaryGetValueIfPresent, CFDictionaryRef};
use core_foundation::number::{CFNumber, CFNumberGetTypeID, CFNumberRef};
use core_foundation::string::{CFString, CFStringGetTypeID, CFStringRef};
use core_foundation::url::CFURL;
use security_framework::os::macos::code_signing::{Flags, SecRequirement, SecStaticCode};
use security_framework_sys::base::errSecSuccess;
use security_framework_sys::certificate::{SecCertificateCopyCommonName, SecCertificateGetTypeID};
use security_framework_sys::code_signing::SecStaticCodeRef;
use security_framework_sys::trust::{SecTrustCopyCertificateChain, SecTrustGetTypeID, SecTrustRef};

use crate::distribution::schema::ReleaseManifestV1;

const DEVELOPER_ID_ISSUER_OID: &str = "1.2.840.113635.100.6.2.6";
const DEVELOPER_ID_APPLICATION_OID: &str = "1.2.840.113635.100.6.1.13";

#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub(super) enum CodeSigningError {
    #[error("the release manifest does not match the compiled code-signing policy")]
    Policy,
    #[error("the staged executable does not satisfy the native code-signing policy")]
    InvalidSignature,
}

pub(super) struct SigningPolicy {
    team_id: String,
    identifier: String,
}

impl SigningPolicy {
    #[cfg(test)]
    pub(super) fn for_test(team_id: &str, identifier: &str) -> Result<Self, CodeSigningError> {
        if team_id.len() != 10
            || !team_id
                .bytes()
                .all(|byte| byte.is_ascii_uppercase() || byte.is_ascii_digit())
            || identifier.is_empty()
            || !identifier
                .bytes()
                .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'.' | b'-'))
        {
            return Err(CodeSigningError::Policy);
        }
        Ok(Self {
            team_id: team_id.to_owned(),
            identifier: identifier.to_owned(),
        })
    }

    pub(super) fn require_manifest(
        &self,
        manifest: &ReleaseManifestV1,
    ) -> Result<(), CodeSigningError> {
        let signing = manifest.code_signing();
        if signing.team_id() != self.team_id || signing.identifier() != self.identifier {
            return Err(CodeSigningError::Policy);
        }
        Ok(())
    }

    pub(super) fn requirement(&self) -> String {
        format!(
            "anchor apple generic and anchor trusted and identifier \"{}\" and certificate 1[field.{}] exists and certificate leaf[field.{}] exists and certificate leaf[subject.OU] = \"{}\"",
            self.identifier,
            DEVELOPER_ID_ISSUER_OID,
            DEVELOPER_ID_APPLICATION_OID,
            self.team_id,
        )
    }
}

#[derive(Clone)]
pub(super) struct SigningInfoView {
    pub(super) identifier_matches: bool,
    pub(super) team_matches: bool,
    pub(super) flags: Option<u32>,
    pub(super) timestamp_is_date: bool,
    pub(super) raw_entitlements_absent: bool,
    pub(super) dictionary_entitlements_absent: bool,
    pub(super) certificate_chain_length: Option<usize>,
    pub(super) leaf_common_name_matches: bool,
}

const SIGNATURE_ADHOC: u32 = 0x0002;
const SIGNATURE_RUNTIME: u32 = 0x1_0000;
const SIGNATURE_LINKER_SIGNED: u32 = 0x2_0000;
const MAX_CERTIFICATE_CHAIN_LENGTH: usize = 8;

const SIGNING_INFORMATION: u32 = 1 << 1;

#[link(name = "Security", kind = "framework")]
extern "C" {
    static kSecCodeInfoEntitlements: CFStringRef;
    static kSecCodeInfoEntitlementsDict: CFStringRef;
    static kSecCodeInfoFlags: CFStringRef;
    static kSecCodeInfoIdentifier: CFStringRef;
    static kSecCodeInfoTeamIdentifier: CFStringRef;
    static kSecCodeInfoTimestamp: CFStringRef;
    static kSecCodeInfoTrust: CFStringRef;

    fn SecCodeCopySigningInformation(
        code: SecStaticCodeRef,
        flags: u32,
        information: *mut CFDictionaryRef,
    ) -> i32;
}

pub(super) fn validate_signing_info(view: &SigningInfoView) -> Result<(), CodeSigningError> {
    let flags = view.flags.ok_or(CodeSigningError::InvalidSignature)?;
    if !view.identifier_matches
        || !view.team_matches
        || flags & SIGNATURE_RUNTIME == 0
        || flags & (SIGNATURE_ADHOC | SIGNATURE_LINKER_SIGNED) != 0
        || !view.timestamp_is_date
        || !view.raw_entitlements_absent
        || !view.dictionary_entitlements_absent
        || !matches!(
            view.certificate_chain_length,
            Some(1..=MAX_CERTIFICATE_CHAIN_LENGTH)
        )
        || !view.leaf_common_name_matches
    {
        return Err(CodeSigningError::InvalidSignature);
    }
    Ok(())
}

pub(super) fn verify_path(
    path: &Path,
    manifest: &ReleaseManifestV1,
    policy: &SigningPolicy,
) -> Result<(), CodeSigningError> {
    policy.require_manifest(manifest)?;
    let url = CFURL::from_path(path, false).ok_or(CodeSigningError::InvalidSignature)?;
    let requirement = SecRequirement::from_str(&policy.requirement())
        .map_err(|_| CodeSigningError::InvalidSignature)?;
    let code = SecStaticCode::from_path(&url, Flags::NONE)
        .map_err(|_| CodeSigningError::InvalidSignature)?;
    let flags = Flags::CHECK_ALL_ARCHITECTURES
        | Flags::STRICT_VALIDATE
        | Flags::CHECK_TRUSTED_ANCHORS
        | Flags::NO_NETWORK_ACCESS;
    code.check_validity(flags, &requirement)
        .map_err(|_| CodeSigningError::InvalidSignature)?;
    let view = copy_signing_info(
        &code,
        &policy.identifier,
        &policy.team_id,
        manifest.code_signing().certificate_common_name(),
    )?;
    validate_signing_info(&view)
}

#[cfg(test)]
pub(super) fn inspect_apple_binary_for_test(
    path: &Path,
) -> Result<SigningInfoView, CodeSigningError> {
    let url = CFURL::from_path(path, false).ok_or(CodeSigningError::InvalidSignature)?;
    let requirement =
        SecRequirement::from_str("anchor apple").map_err(|_| CodeSigningError::InvalidSignature)?;
    let code = SecStaticCode::from_path(&url, Flags::NONE)
        .map_err(|_| CodeSigningError::InvalidSignature)?;
    let flags = Flags::CHECK_ALL_ARCHITECTURES | Flags::STRICT_VALIDATE | Flags::NO_NETWORK_ACCESS;
    code.check_validity(flags, &requirement)
        .map_err(|_| CodeSigningError::InvalidSignature)?;
    copy_signing_info(
        &code,
        "test.invalid.identifier",
        "ZZZZZZZZZZ",
        "test invalid leaf common name",
    )
}

fn copy_signing_info(
    code: &SecStaticCode,
    identifier: &str,
    team_id: &str,
    leaf_common_name: &str,
) -> Result<SigningInfoView, CodeSigningError> {
    let mut raw_dictionary = ptr::null();
    // SAFETY: `code` is a live SecStaticCode. The Security framework writes a
    // retained CFDictionary to the non-null output pointer on success.
    let status = unsafe {
        SecCodeCopySigningInformation(
            code.as_concrete_TypeRef(),
            SIGNING_INFORMATION,
            &mut raw_dictionary,
        )
    };
    if status != errSecSuccess || raw_dictionary.is_null() {
        return Err(CodeSigningError::InvalidSignature);
    }
    // SAFETY: successful SecCodeCopySigningInformation follows the create
    // rule. This wrapper is therefore the dictionary's sole owned reference.
    let dictionary = unsafe {
        CFDictionary::<*const c_void, *const c_void>::wrap_under_create_rule(raw_dictionary)
    };
    let identifier = CFString::new(identifier);
    let team_id = CFString::new(team_id);
    let leaf_common_name = CFString::new(leaf_common_name);
    let (certificate_chain_length, leaf_common_name_matches) =
        certificate_chain_evidence(&dictionary, &leaf_common_name);
    Ok(SigningInfoView {
        identifier_matches: string_value_matches(
            &dictionary,
            unsafe { kSecCodeInfoIdentifier },
            &identifier,
        ),
        team_matches: string_value_matches(
            &dictionary,
            unsafe { kSecCodeInfoTeamIdentifier },
            &team_id,
        ),
        flags: number_value(&dictionary, unsafe { kSecCodeInfoFlags }),
        timestamp_is_date: typed_value(&dictionary, unsafe { kSecCodeInfoTimestamp }, unsafe {
            CFDateGetTypeID()
        })
        .is_some(),
        raw_entitlements_absent: dictionary_value(&dictionary, unsafe { kSecCodeInfoEntitlements })
            .is_none(),
        dictionary_entitlements_absent: dictionary_value(&dictionary, unsafe {
            kSecCodeInfoEntitlementsDict
        })
        .is_none(),
        certificate_chain_length,
        leaf_common_name_matches,
    })
}

fn dictionary_value(
    dictionary: &CFDictionary<*const c_void, *const c_void>,
    key: CFStringRef,
) -> Option<CFTypeRef> {
    if key.is_null() {
        return None;
    }
    let mut value: *const c_void = ptr::null();
    // SAFETY: both references are live Core Foundation objects and the output
    // pointer is valid for this call. The returned value is borrowed from the
    // dictionary and is never released here.
    let found = unsafe {
        CFDictionaryGetValueIfPresent(dictionary.as_concrete_TypeRef(), key.cast(), &mut value)
    };
    (found != 0 && !value.is_null()).then_some(value.cast())
}

fn typed_value(
    dictionary: &CFDictionary<*const c_void, *const c_void>,
    key: CFStringRef,
    expected_type: CFTypeID,
) -> Option<CFTypeRef> {
    let value = dictionary_value(dictionary, key)?;
    // SAFETY: dictionary_value excludes null and returns a borrowed CF object.
    (unsafe { CFGetTypeID(value) } == expected_type).then_some(value)
}

fn string_value_matches(
    dictionary: &CFDictionary<*const c_void, *const c_void>,
    key: CFStringRef,
    expected: &CFString,
) -> bool {
    let Some(value) = typed_value(dictionary, key, unsafe { CFStringGetTypeID() }) else {
        return false;
    };
    // SAFETY: both values are live Core Foundation objects. `value` was
    // type-checked as a CFString immediately above.
    unsafe { CFEqual(value, expected.as_CFTypeRef()) != 0 }
}

fn number_value(
    dictionary: &CFDictionary<*const c_void, *const c_void>,
    key: CFStringRef,
) -> Option<u32> {
    let value = typed_value(dictionary, key, unsafe { CFNumberGetTypeID() })?;
    // SAFETY: `value` was type-checked as a CFNumber and remains retained by
    // the dictionary for the lifetime of this wrapper.
    let number = unsafe { CFNumber::wrap_under_get_rule(value as CFNumberRef) };
    u32::try_from(number.to_i64()?).ok()
}

fn certificate_chain_evidence(
    dictionary: &CFDictionary<*const c_void, *const c_void>,
    expected_leaf_common_name: &CFString,
) -> (Option<usize>, bool) {
    let Some(trust) = typed_value(dictionary, unsafe { kSecCodeInfoTrust }, unsafe {
        SecTrustGetTypeID()
    }) else {
        return (None, false);
    };
    // SAFETY: `trust` was type-checked as SecTrust and CopySigningInformation
    // returned it after successful code validity evaluation.
    let raw_chain = unsafe { SecTrustCopyCertificateChain(trust as SecTrustRef) };
    if raw_chain.is_null() {
        return (None, false);
    }
    // SAFETY: SecTrustCopyCertificateChain follows the create rule.
    let chain = unsafe { CFArray::<*const c_void>::wrap_under_create_rule(raw_chain) };
    let Ok(length) = usize::try_from(chain.len()) else {
        return (None, false);
    };
    if !(1..=MAX_CERTIFICATE_CHAIN_LENGTH).contains(&length) {
        return (Some(length), false);
    }
    for index in 0..length {
        // SAFETY: `index` is bounded by the checked array length.
        let certificate =
            unsafe { CFArrayGetValueAtIndex(chain.as_concrete_TypeRef(), index as isize) }
                as CFTypeRef;
        if certificate.is_null()
            || unsafe { CFGetTypeID(certificate) } != unsafe { SecCertificateGetTypeID() }
        {
            return (Some(length), false);
        }
    }
    // SAFETY: the checked nonempty array remains alive for this scope.
    let leaf = unsafe { CFArrayGetValueAtIndex(chain.as_concrete_TypeRef(), 0) } as CFTypeRef;
    if leaf.is_null() || unsafe { CFGetTypeID(leaf) } != unsafe { SecCertificateGetTypeID() } {
        return (Some(length), false);
    }
    let mut common_name = ptr::null();
    // SAFETY: `leaf` is a type-checked SecCertificate and the output pointer is
    // valid. A successful call returns a retained CFString.
    let status = unsafe { SecCertificateCopyCommonName(leaf as _, &mut common_name) };
    if status != errSecSuccess || common_name.is_null() {
        return (Some(length), false);
    }
    // SAFETY: successful SecCertificateCopyCommonName follows the create rule.
    let common_name = unsafe { CFString::wrap_under_create_rule(common_name) };
    let matches = unsafe {
        CFEqual(
            common_name.as_CFTypeRef(),
            expected_leaf_common_name.as_CFTypeRef(),
        ) != 0
    };
    (Some(length), matches)
}
