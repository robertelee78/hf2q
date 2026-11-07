//! Routed-expert prefill schedule reuse (gap program fix 2).
//!
//! A DeepSeek-V4 prefill FFN builds the compact `mm_id` routing schedule
//! (`map0c`) once for the paired IQ2_XXS gate/up call and, today, again for
//! the slotted Q2_K down call. Both calls route the same `[rows, top_k]`
//! expert ids in the same encoder, so the down call can consume the gate/up
//! schedule (`quantized_matmul_id_ggml_pooled_slotted_reusing_schedule`):
//! one `map0c` per layer instead of two, byte-identical output (the schedule
//! depends only on the ids). mlx-native refuses a schedule recorded for other
//! ids, shapes or another encoder before encoding anything.
//!
//! `HF2Q_DEEPSEEK_MOE_DOWN_REUSE` (read once at model load, default off).

use anyhow::Result;

/// Serving switch for the down projection's schedule reuse.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum MoeDownScheduleReuse {
    /// The down call builds its own schedule.
    #[default]
    Off,
    /// The down call reuses the paired gate/up call's schedule.
    On,
}

impl MoeDownScheduleReuse {
    /// Process switch read by [`Self::from_env`].
    pub const ENV: &'static str = "HF2Q_DEEPSEEK_MOE_DOWN_REUSE";

    /// `HF2Q_DEEPSEEK_MOE_DOWN_REUSE`: `1` selects [`Self::On`], `0` or unset
    /// [`Self::Off`]; anything else is an error.
    pub fn from_env() -> Result<Self> {
        Self::resolve(std::env::var_os(Self::ENV).as_deref())
    }

    /// Resolution rule of [`Self::from_env`].
    pub fn resolve(value: Option<&std::ffi::OsStr>) -> Result<Self> {
        match value.map(|value| value.to_str()) {
            None | Some(Some("0")) => Ok(Self::Off),
            Some(Some("1")) => Ok(Self::On),
            Some(_) => anyhow::bail!("{} must be 0 or 1 (got {value:?})", Self::ENV),
        }
    }

    pub fn enabled(self) -> bool {
        self == Self::On
    }
}

#[cfg(test)]
mod tests {
    use std::ffi::OsStr;

    use super::*;

    #[test]
    fn switch_parses_zero_one_and_unset_and_rejects_anything_else() {
        assert_eq!(MoeDownScheduleReuse::resolve(None).unwrap(), MoeDownScheduleReuse::Off);
        assert_eq!(
            MoeDownScheduleReuse::resolve(Some(OsStr::new("0"))).unwrap(),
            MoeDownScheduleReuse::Off
        );
        assert_eq!(
            MoeDownScheduleReuse::resolve(Some(OsStr::new("1"))).unwrap(),
            MoeDownScheduleReuse::On
        );
        for bad in ["", "2", "on", "true"] {
            assert!(MoeDownScheduleReuse::resolve(Some(OsStr::new(bad))).is_err(), "{bad:?}");
        }
        assert!(!MoeDownScheduleReuse::default().enabled());
    }
}
