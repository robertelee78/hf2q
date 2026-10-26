//! Explicit default-OFF experiment; invalid intent never becomes legacy success.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Mode {
    Off,
    Selected,
    Refused,
    Invalid,
}
impl Mode {
    pub fn from_os(value: Option<&std::ffi::OsStr>, ack: bool) -> Self {
        match value {
            None => Self::Off,
            Some(v) if v == "1" && ack => Self::Selected,
            Some(v) if v == "1" => Self::Refused,
            Some(_) => Self::Invalid,
        }
    }
    pub fn validate(self) -> Result<bool, &'static str> {
        match self {
            Self::Off => Ok(false),
            Self::Selected => Ok(true),
            Self::Refused => {
                Err("HF2Q_GEMMA_FULL_INPUT_CHECKPOINT requires HF2Q_UNSAFE_EXPERIMENTS=1")
            }
            Self::Invalid => Err("HF2Q_GEMMA_FULL_INPUT_CHECKPOINT must be 1 or unset"),
        }
    }
    pub fn summary(self) -> Option<&'static str> {
        match self {
            Self::Off => None,
            Self::Selected => Some("UNSAFE (ack-required, activated): HF2Q_GEMMA_FULL_INPUT_CHECKPOINT=1; experimental Hybrid input checkpoint, CPU requested-anchor quota1536MiB; NOT full-prefix qualification"),
            Self::Refused => Some("REFUSED HF2Q_GEMMA_FULL_INPUT_CHECKPOINT=1: acknowledgment missing"),
            Self::Invalid => Some("REFUSED HF2Q_GEMMA_FULL_INPUT_CHECKPOINT: invalid value"),
        }
    }
}
pub fn selected() -> Result<bool, &'static str> {
    super::INVESTIGATION_ENV
        .gemma_full_input_checkpoint
        .validate()
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn gemma_full_input_activation_matrix_is_explicit() {
        assert_eq!(Mode::from_os(None, false).validate(), Ok(false));
        assert_eq!(Mode::from_os(Some("1".as_ref()), true).validate(), Ok(true));
        assert!(Mode::from_os(Some("1".as_ref()), false).validate().is_err());
        for value in ["0", "true", "", "other"] {
            assert!(Mode::from_os(Some(value.as_ref()), true)
                .validate()
                .is_err());
        }
    }
}
