//! Strict startup intent for the optional Category-3 pool policy.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum QwenPoolProfile {
    Off,
    Selected,
    Refused,
    Invalid,
}
/// Match the existing split-profile branch's exact equality-to-1 selector.
pub fn split_profile_conflict(value: Option<&str>) -> bool {
    value == Some("1")
}
impl QwenPoolProfile {
    pub fn parse(value: Option<&str>, ack: bool) -> Self {
        match value {
            None => Self::Off,
            Some("adr063-24g-v1") if ack => Self::Selected,
            Some("adr063-24g-v1") => Self::Refused,
            Some(_) => Self::Invalid,
        }
    }
    pub fn from_os(value: Option<&std::ffi::OsStr>, ack: bool) -> Self {
        match value {
            None => Self::Off,
            Some(v) => match v.to_str() {
                Some(v) => Self::parse(Some(v), ack),
                None => Self::Invalid,
            },
        }
    }
    pub fn validate(self) -> Result<bool, &'static str> {
        match self {
            Self::Off => Ok(false),
            Self::Selected => Ok(true),
            Self::Refused => Err(
                "Qwen pool profile requires HF2Q_UNSAFE_EXPERIMENTS=1; refusing uncapped fallback",
            ),
            Self::Invalid => {
                Err("unknown HF2Q_QWEN_POOL_GROWTH_PROFILE; expected adr063-24g-v1 or unset")
            }
        }
    }
    pub fn summary(self) -> Option<&'static str> {
        match self {
            Self::Off => None,
            Self::Selected => Some("Qwen pool adr063-24g-v1 requested: experimental enforcing policy, NOT whole-memory admission"),
            Self::Refused => Some("REFUSED Qwen pool profile: acknowledgment missing; serve startup must fail"),
            Self::Invalid => Some("REFUSED Qwen pool profile: invalid value; serve startup must fail"),
        }
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn qwen_pool_split_profile_conflict_matches_actual_selector() {
        assert!(split_profile_conflict(Some("1")));
        for value in [None, Some("0"), Some("true"), Some("")] {
            assert!(!split_profile_conflict(value));
        }
    }
    #[test]
    fn qwen_pool_nonunicode_intent_is_not_off() {
        use std::os::unix::ffi::OsStrExt;
        let value = std::ffi::OsStr::from_bytes(&[255]);
        assert_eq!(
            QwenPoolProfile::from_os(Some(value), true),
            QwenPoolProfile::Invalid
        );
    }
    #[test]
    fn qwen_pool_intent_requires_exact_profile_and_ack() {
        assert_eq!(QwenPoolProfile::parse(None, false), QwenPoolProfile::Off);
        assert_eq!(QwenPoolProfile::parse(None, true), QwenPoolProfile::Off);
        assert!(QwenPoolProfile::parse(Some("adr063-24g-v1"), true)
            .validate()
            .unwrap());
        assert!(QwenPoolProfile::parse(Some("adr063-24g-v1"), false)
            .validate()
            .is_err());
        for value in ["", "0", "1", "adr063-24g-v2", "adr063-24g-v1 "] {
            assert!(QwenPoolProfile::parse(Some(value), true)
                .validate()
                .is_err());
        }
    }
}
