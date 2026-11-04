//! ADR-063 Q4-GEMMA-SPEC-SERVE — Gemma 4 serving decode telemetry (§5.9).
//!
//! Per-width decode tick counters. In a speculation-off run they give the
//! time-weighted width histogram `h(n) = tick_seconds(n) / Σ_m
//! tick_seconds(m)` that the M0/G0 projection (§8.1) needs.

use std::fmt::Write as _;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

/// Widest decode tick tracked per bucket; wider ticks land in the last
/// bucket, which is rendered as `width="64+"`.
const MAX_WIDTH: usize = 64;

pub(crate) struct GemmaDecodeTelemetry {
    ticks: [[AtomicU64; MAX_WIDTH + 1]; 2],
    nanos: [[AtomicU64; MAX_WIDTH + 1]; 2],
}

#[allow(clippy::declare_interior_mutable_const)]
const ZERO: AtomicU64 = AtomicU64::new(0);
#[allow(clippy::declare_interior_mutable_const)]
const ROW: [AtomicU64; MAX_WIDTH + 1] = [ZERO; MAX_WIDTH + 1];

pub(crate) static TELEMETRY: GemmaDecodeTelemetry = GemmaDecodeTelemetry {
    ticks: [ROW; 2],
    nanos: [ROW; 2],
};

impl GemmaDecodeTelemetry {
    pub(crate) fn record(&self, width: usize, speculative: bool, nanos: u64) {
        if width == 0 {
            return;
        }
        let bucket = width.min(MAX_WIDTH);
        let arm = usize::from(speculative);
        self.ticks[arm][bucket].fetch_add(1, Ordering::Relaxed);
        self.nanos[arm][bucket].fetch_add(nanos, Ordering::Relaxed);
    }

    /// Prometheus text for `hf2q_gemma_decode_ticks_by_width_total` and
    /// `hf2q_gemma_decode_tick_seconds_total`, nonzero buckets only.
    pub(crate) fn render_prometheus(&self) -> String {
        let mut ticks = String::from(
            "# HELP hf2q_gemma_decode_ticks_by_width_total Gemma 4 serving decode ticks by decoding width.\n\
# TYPE hf2q_gemma_decode_ticks_by_width_total counter\n",
        );
        let mut seconds = String::from(
            "# HELP hf2q_gemma_decode_tick_seconds_total Gemma 4 serving decode tick wall seconds by decoding width.\n\
# TYPE hf2q_gemma_decode_tick_seconds_total counter\n",
        );
        for (arm, label) in [(0usize, "false"), (1, "true")] {
            for bucket in 1..=MAX_WIDTH {
                let count = self.ticks[arm][bucket].load(Ordering::Relaxed);
                if count == 0 {
                    continue;
                }
                let nanos = self.nanos[arm][bucket].load(Ordering::Relaxed);
                let width = if bucket == MAX_WIDTH {
                    format!("{MAX_WIDTH}+")
                } else {
                    bucket.to_string()
                };
                let _ = writeln!(
                    ticks,
                    "hf2q_gemma_decode_ticks_by_width_total{{width=\"{width}\",speculative=\"{label}\"}} {count}"
                );
                let _ = writeln!(
                    seconds,
                    "hf2q_gemma_decode_tick_seconds_total{{width=\"{width}\",speculative=\"{label}\"}} {:.9}",
                    nanos as f64 / 1e9
                );
            }
        }
        ticks + &seconds
    }
}

/// Records one decode tick's wall time on drop.
pub(crate) struct GemmaDecodeTickTimer {
    width: usize,
    speculative: bool,
    start: Instant,
}

impl GemmaDecodeTickTimer {
    pub(crate) fn start(width: usize, speculative: bool) -> Self {
        Self {
            width,
            speculative,
            start: Instant::now(),
        }
    }
}

impl Drop for GemmaDecodeTickTimer {
    fn drop(&mut self) {
        TELEMETRY.record(
            self.width,
            self.speculative,
            self.start.elapsed().as_nanos() as u64,
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gemma_decode_width_telemetry_renders_nonzero_buckets() {
        let telemetry = GemmaDecodeTelemetry {
            ticks: [ROW; 2],
            nanos: [ROW; 2],
        };
        assert_eq!(telemetry.render_prometheus().lines().count(), 4);
        telemetry.record(4, false, 22_000_000);
        telemetry.record(4, false, 23_000_000);
        telemetry.record(1, false, 9_000_000);
        telemetry.record(0, false, 5);
        telemetry.record(200, true, 1_000_000_000);
        let text = telemetry.render_prometheus();
        assert!(text.contains(
            "hf2q_gemma_decode_ticks_by_width_total{width=\"4\",speculative=\"false\"} 2\n"
        ));
        assert!(text.contains(
            "hf2q_gemma_decode_tick_seconds_total{width=\"4\",speculative=\"false\"} 0.045000000\n"
        ));
        assert!(text.contains(
            "hf2q_gemma_decode_ticks_by_width_total{width=\"1\",speculative=\"false\"} 1\n"
        ));
        assert!(text.contains(
            "hf2q_gemma_decode_ticks_by_width_total{width=\"64+\",speculative=\"true\"} 1\n"
        ));
        assert!(!text.contains("width=\"2\""));
    }
}
