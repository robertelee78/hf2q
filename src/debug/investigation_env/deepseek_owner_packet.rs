//! Closed default-OFF selected DeepSeek research diagnostic intent.
use std::{
    ffi::OsStr,
    path::{Path, PathBuf},
};
#[derive(Debug, Clone)]
pub enum Mode {
    Off,
    Selected(PathBuf),
    Refused,
    Invalid,
}
impl Mode {
    pub fn from_os(value: Option<&OsStr>, ack: bool) -> Self {
        match value {
            None => Self::Off,
            Some(value)
                if value.is_empty()
                    || value.as_encoded_bytes().len() > 4096
                    || !Path::new(value).is_absolute() =>
            {
                Self::Invalid
            }
            Some(_) if !ack => Self::Refused,
            Some(value) => Self::Selected(PathBuf::from(value)),
        }
    }
    pub fn validate(&self) -> Result<Option<&Path>, &'static str> {
        match self {
            Self::Off => Ok(None),
            Self::Selected(path) => Ok(Some(path)),
            Self::Refused => Err("HF2Q_DEEPSEEK_OWNER_PACKET requires HF2Q_UNSAFE_EXPERIMENTS=1"),
            Self::Invalid => {
                Err("HF2Q_DEEPSEEK_OWNER_PACKET requires a bounded absolute packet path or unset")
            }
        }
    }
    pub fn summary(&self) -> Option<&'static str> {
        match self {
            Self::Off => None,
            Self::Selected(_) => Some("UNSAFE (ack-required, activated): HF2Q_DEEPSEEK_OWNER_PACKET; selected DeepSeek research collector; default packet refuses preload; separately reviewed capped operating case required; no physical/Q5 qualification authority"),
            Self::Refused => Some("REFUSED HF2Q_DEEPSEEK_OWNER_PACKET: acknowledgment missing"),
            Self::Invalid => Some("REFUSED HF2Q_DEEPSEEK_OWNER_PACKET: invalid path"),
        }
    }
}
pub fn selected() -> Result<Option<&'static Path>, &'static str> {
    super::INVESTIGATION_ENV.deepseek_owner_packet.validate()
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn actual_deepseek_owner_activation_matrix() {
        assert!(Mode::from_os(None, false).validate().unwrap().is_none());
        assert_eq!(
            Mode::from_os(Some(OsStr::new("/tmp/packet.json")), true)
                .validate()
                .unwrap(),
            Some(Path::new("/tmp/packet.json"))
        );
        assert!(Mode::from_os(Some(OsStr::new("/tmp/packet.json")), false)
            .validate()
            .is_err());
        for bad in ["", "relative.json", "0", "1"] {
            assert!(Mode::from_os(Some(OsStr::new(bad)), true)
                .validate()
                .is_err());
        }
    }
}
