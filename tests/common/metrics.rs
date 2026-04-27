//! ADR-014 P10 iter-1 — `RunMetrics` is the single observation type
//! emitted by every subprocess wrapper in [`super`].
//!
//! Schema (Decision 15 columns the harness ingests):
//! - `wall_s`: median of three timed runs in seconds (Decision 15
//!   warmup-discarded + 60 s thermal cooldown + 3 timed; sorted; median
//!   = row 2 of 3). Sentinel `-1.0` indicates the binary or Python
//!   module was missing — see [`RunMetrics::missing_binary`].
//! - `peak_rss_bytes`: maximum resident-set-size in bytes, parsed from
//!   `/usr/bin/time -l` BSD output (`maximum resident set size`). On
//!   macOS this is bytes; on Linux GNU `time -v` reports KB and would
//!   need a separate parser — out of scope for this iter (peers we
//!   benchmark on macOS M5 Max). Sentinel `u64::MAX` for missing.
//! - `exit_code`: subprocess exit status; `-1` for missing-binary.
//! - `stderr_tail`: last ~4 KiB of stderr (truncated to keep the
//!   markdown table readable). For `missing_binary`, contains a
//!   `missing: <name>` description string.
//!
//! No `.unwrap()` outside test code. Sentinel constructor is `pub`
//! for use by the runner modules and the harness.

/// Immutable record of a subprocess run (one of the three timed
/// invocations after warmup + cooldown, or a missing-binary sentinel).
#[derive(Debug, Clone, PartialEq)]
pub struct RunMetrics {
    pub wall_s: f64,
    pub peak_rss_bytes: u64,
    pub exit_code: i32,
    pub stderr_tail: String,
}

impl RunMetrics {
    /// Constructs the canonical missing-binary sentinel. Used by every
    /// runner when `which`-equivalent lookup fails or when the Python
    /// import returns `ImportError`. The harness recognises this by
    /// `wall_s == -1.0` and emits `Verdict::NotMeasured`.
    ///
    /// `name` is the absent binary or module label (e.g.
    /// `"llama-quantize"`, `"mlx_lm"`); it is recorded verbatim into
    /// `stderr_tail` for the markdown table.
    pub fn missing_binary(name: &str) -> Self {
        Self {
            wall_s: -1.0,
            peak_rss_bytes: u64::MAX,
            exit_code: -1,
            stderr_tail: format!("missing: {name}"),
        }
    }

    /// True when this record is the missing-binary sentinel rather
    /// than a real observation. The harness uses this to route the
    /// cell to `Verdict::NotMeasured`.
    pub fn is_missing_binary(&self) -> bool {
        self.wall_s == -1.0 && self.peak_rss_bytes == u64::MAX && self.exit_code == -1
    }
}

impl Default for RunMetrics {
    /// Default = the canonical missing-binary sentinel with an
    /// `unknown` label. Real wrappers always use
    /// [`RunMetrics::missing_binary`] with a meaningful name; this
    /// `Default` impl exists only so `RunMetrics::default()` compiles
    /// for callers that need a placeholder.
    fn default() -> Self {
        Self::missing_binary("unknown")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn missing_binary_sentinel_has_documented_field_values() {
        let m = RunMetrics::missing_binary("llama-quantize");
        assert_eq!(m.wall_s, -1.0);
        assert_eq!(m.peak_rss_bytes, u64::MAX);
        assert_eq!(m.exit_code, -1);
        assert_eq!(m.stderr_tail, "missing: llama-quantize");
        assert!(m.is_missing_binary());
    }

    #[test]
    fn default_is_the_missing_binary_sentinel() {
        let d = RunMetrics::default();
        assert!(d.is_missing_binary());
        assert_eq!(d.stderr_tail, "missing: unknown");
    }

    #[test]
    fn real_observation_is_not_classified_as_missing() {
        let real = RunMetrics {
            wall_s: 1.234,
            peak_rss_bytes: 1024 * 1024,
            exit_code: 0,
            stderr_tail: String::new(),
        };
        assert!(!real.is_missing_binary());
    }
}
