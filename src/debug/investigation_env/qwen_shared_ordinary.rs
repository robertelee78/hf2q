//! Closed startup intent for the experimental owning-worker Q38 shared target.
#[derive(Clone,Copy,Debug,PartialEq,Eq)]
pub enum Mode { Off,On,Refused,Invalid }
impl Mode {
    pub fn from_os(value:Option<&std::ffi::OsStr>,ack:bool)->Self {
        match value.and_then(|x|x.to_str()) {
            None if value.is_none()=>Self::Off,
            Some("0")=>Self::Off,
            Some("1") if ack=>Self::On,
            Some("1")=>Self::Refused,
            _=>Self::Invalid,
        }
    }
    pub fn validate(self)->Result<bool,&'static str> {
        match self {
            Self::Off=>Ok(false),Self::On=>Ok(true),
            Self::Refused=>Err("HF2Q_QWEN_SHARED_ORDINARY=1 requires HF2Q_UNSAFE_EXPERIMENTS=1"),
            Self::Invalid=>Err("HF2Q_QWEN_SHARED_ORDINARY expects unset,0,or1; refusing fallback"),
        }
    }
    pub fn summary(self)->Option<&'static str> {
        match self { Self::Off=>None,Self::On=>Some("Qwen shared ordinary target requested: experimental, not whole-body B4/performance admission"),
            _=>Some("REFUSED invalid/unacknowledged Qwen shared ordinary target intent") }
    }
}
#[cfg(test)]mod controls {
    use super::*;
    #[test]fn closed_intent_requires_ack() {
        assert_eq!(Mode::from_os(None,false).validate(),Ok(false));
        assert_eq!(Mode::from_os(Some("0".as_ref()),false).validate(),Ok(false));
        assert!(Mode::from_os(Some("1".as_ref()),false).validate().is_err());
        assert_eq!(Mode::from_os(Some("1".as_ref()),true).validate(),Ok(true));
        for x in ["", "true","2","adr063"] {assert!(Mode::from_os(Some(x.as_ref()),true).validate().is_err());}
    }
    #[test]fn nonunicode_is_invalid_not_off() {
        use std::os::unix::ffi::OsStrExt;
        assert!(Mode::from_os(Some(std::ffi::OsStr::from_bytes(&[255])),true).validate().is_err());
    }
}
