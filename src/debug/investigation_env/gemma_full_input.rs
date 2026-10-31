//! Gemma full-input checkpoint activation (ADR-063). Default ON for the
//! supported Gemma Hybrid/TQ8 SlotAware 1–4 substrate; `=0` is the kill
//! switch, `=1` forces it and refuses an unsupported substrate. Invalid
//! intent never silently becomes a weaker contract.
use std::cell::Cell;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Mode {
    /// Unset: on exactly when the substrate is supported, otherwise off.
    Default,
    /// `=1`: on; an unsupported substrate refuses startup.
    On,
    /// `=0`: kill switch.
    Off,
    Invalid,
}

impl Mode {
    pub fn from_os(value: Option<&std::ffi::OsStr>) -> Self {
        match value {
            None => Self::Default,
            Some(v) if v == "1" => Self::On,
            Some(v) if v == "0" => Self::Off,
            Some(_) => Self::Invalid,
        }
    }

    pub fn validate(self) -> Result<(), &'static str> {
        match self {
            Self::Invalid => Err("HF2Q_GEMMA_FULL_INPUT_CHECKPOINT must be 0, 1 or unset"),
            Self::Default | Self::On | Self::Off => Ok(()),
        }
    }

    /// `supported` covers family, scheduler, KV regime, codebook and the
    /// four-slot CPU anchor bound for the actual provisioned geometry.
    pub fn decide(self, supported: bool) -> Result<bool, &'static str> {
        match self {
            Self::Default => Ok(supported),
            Self::On if supported => Ok(true),
            Self::On => Err(
                "HF2Q_GEMMA_FULL_INPUT_CHECKPOINT=1 requires Gemma Hybrid/TQ8 SlotAware 1–4 \
                 within the CPU anchor quota; no fallback",
            ),
            Self::Off => Ok(false),
            Self::Invalid => Err("HF2Q_GEMMA_FULL_INPUT_CHECKPOINT must be 0, 1 or unset"),
        }
    }

    pub fn summary(self) -> Option<&'static str> {
        match self {
            Self::Default => None,
            Self::On => Some("HF2Q_GEMMA_FULL_INPUT_CHECKPOINT=1: Gemma full-input checkpoint forced on; unsupported substrate refuses startup"),
            Self::Off => Some("HF2Q_GEMMA_FULL_INPUT_CHECKPOINT=0: Gemma full-input checkpoint disabled (kill switch)"),
            Self::Invalid => Some("REFUSED HF2Q_GEMMA_FULL_INPUT_CHECKPOINT: invalid value (use 0, 1 or unset)"),
        }
    }
}

pub fn mode() -> Mode {
    super::INVESTIGATION_ENV.gemma_full_input_checkpoint
}

/// Process-start validation: only an invalid value refuses.
pub fn validate() -> Result<(), &'static str> {
    mode().validate()
}

thread_local! {
    /// Set once by the Gemma SlotAware worker thread that owns the Hybrid
    /// scaffold. Anchor quota and fallible CPU copies apply only there, so
    /// another engine in the same process keeps its own policy.
    static WORKER_ACTIVE: Cell<bool> = const { Cell::new(false) };
}

pub fn worker_active() -> bool {
    WORKER_ACTIVE.with(Cell::get)
}

pub fn set_worker_active(active: bool) {
    WORKER_ACTIVE.with(|flag| flag.set(active));
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gemma_full_input_default_on_kill_switch_and_forced_matrix() {
        let parse = |value: Option<&str>| Mode::from_os(value.map(std::ffi::OsStr::new));
        assert_eq!(parse(None), Mode::Default);
        assert_eq!(parse(Some("1")), Mode::On);
        assert_eq!(parse(Some("0")), Mode::Off);
        for value in ["", "true", "on", "2", " 1"] {
            assert_eq!(parse(Some(value)), Mode::Invalid, "{value:?}");
            assert!(parse(Some(value)).validate().is_err());
            assert!(parse(Some(value)).decide(true).is_err());
        }
        assert_eq!(Mode::Default.decide(true), Ok(true));
        assert_eq!(
            Mode::Default.decide(false),
            Ok(false),
            "default never refuses"
        );
        assert_eq!(Mode::On.decide(true), Ok(true));
        assert!(
            Mode::On.decide(false).is_err(),
            "forced on never falls back"
        );
        assert_eq!(Mode::Off.decide(true), Ok(false), "kill switch wins");
        assert_eq!(Mode::Off.decide(false), Ok(false));
        assert!(Mode::Default.summary().is_none());
    }
}
