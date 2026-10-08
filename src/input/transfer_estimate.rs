//! Smoothed, bounded progress estimation for model-sized Hub transfers.
//!
//! Issue #249: native Xet (`hf-hub` 1.0 / `hf-xet` 1.5.3) reconstructs a file
//! from many concurrently fetched chunk ranges, buffers them in memory, and
//! reports only *completed, written* logical bytes. `hf-hub`'s poller forwards
//! `GroupProgressReport::total_bytes_completed` and the matching 10-second
//! completion rate; the report's network `total_transfer_bytes_completed` is
//! not forwarded and the download group is crate-private, so hf2q cannot see
//! per-process received bytes without forking the dependency.
//!
//! Written bytes therefore arrive as bursts separated by tens of seconds of
//! no visible change while the network is busy. Xet's 10-second window rate
//! decays toward zero during those flat periods, which produced ETAs from
//! hours to thousands of years. This estimator replaces that with:
//!
//! * a 60-second byte-weighted window rate, smoothed by a time-constant EWMA
//!   and frozen while written bytes are idle, so one flat period cannot swing
//!   the ETA;
//! * an explicit `estimating` state until the rate has warmed up;
//! * an ETA cap, so nothing ever renders years;
//! * a host-wide network receive counter (the only received-bytes signal hf2q
//!   can reach) used as a live activity indicator and to tell Xet buffering
//!   apart from a genuine stall.

use std::collections::VecDeque;
use std::time::Duration;

use serde::{Deserialize, Serialize};

/// No written-byte advance for this long is reported as idle (buffering when
/// the host network is busy, stalled when it is quiet).
pub(crate) const IDLE_AFTER: Duration = Duration::from_secs(15);
/// Byte-weighted rate window. Observed Xet bursts are up to ~60 s apart, so a
/// window this long always spans at least one burst while transferring.
const RATE_WINDOW: Duration = Duration::from_secs(60);
/// Time constant of the EWMA applied on top of the window rate.
const RATE_TIME_CONSTANT: Duration = Duration::from_secs(10);
/// Minimum observation span after the first written byte before a rate is
/// considered stable enough to print.
const WARMUP: Duration = Duration::from_secs(10);
/// Minimum spacing between retained rate samples.
const SAMPLE_SPACING: Duration = Duration::from_secs(1);
/// ETAs are capped here; a longer estimate renders as "more than 24h".
pub(crate) const MAX_ETA_SECONDS: u64 = 24 * 60 * 60;
/// An ETA number is withheld ("estimating") until the transfer has run at
/// least this long ...
const ETA_MIN_ELAPSED: Duration = Duration::from_secs(30);
/// ... has written at least this fraction of the payload, or has run for
/// [`ETA_SLOW_ELAPSED`] (so very slow transfers still get an estimate) ...
const ETA_MIN_FRACTION: f64 = 0.03;
const ETA_SLOW_ELAPSED: Duration = Duration::from_secs(120);
/// ... and the last [`STABILITY_SAMPLES`] window rates, taken at least
/// [`STABILITY_SPACING`] apart while bytes were being written, agree within
/// [`STABILITY_TOLERANCE`] (max/min). Xet ramps concurrency up during a
/// download, so an accelerating transfer keeps showing "estimating" rather
/// than an ETA that is several times too long (#249 field test).
const STABILITY_SAMPLES: usize = 4;
const STABILITY_SPACING: Duration = Duration::from_secs(10);
const STABILITY_TOLERANCE: f64 = 1.5;
/// Once shown, the ETA may change by at most a factor of two per this
/// interval (after counting down elapsed time).
const ETA_DOUBLING_INTERVAL: Duration = Duration::from_secs(5);
/// Host receive rate at or above this counts as network activity.
const NETWORK_ACTIVE_FLOOR: f64 = 256.0 * 1024.0;
/// Time constant for the host receive rate.
const HOST_RATE_TIME_CONSTANT: Duration = Duration::from_secs(3);

#[derive(Clone, Copy, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum TransferState {
    /// Not enough observations for a stable rate yet.
    #[default]
    Estimating,
    /// Written bytes are advancing and the rate is stable.
    Transferring,
    /// No written bytes for [`IDLE_AFTER`], but the host network is receiving.
    Buffering,
    /// No written bytes for [`IDLE_AFTER`] and no host network activity (or no
    /// host counter available).
    Stalled,
}

/// Foreground-safe, wire-safe summary of one transfer at one instant.
#[derive(Clone, Copy, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct TransferEstimate {
    pub(crate) state: TransferState,
    /// Smoothed written-byte rate; `None` while estimating.
    pub(crate) bytes_per_second: Option<u64>,
    /// Bounded ETA; `None` while estimating or stalled.
    pub(crate) eta_seconds: Option<u64>,
    /// Time since written bytes last advanced (or since the transfer began).
    pub(crate) idle_ms: u64,
    /// Host-wide bytes received since the transfer began (all non-loopback
    /// interfaces; includes unrelated traffic).
    pub(crate) host_received_bytes: Option<u64>,
    /// Smoothed host-wide receive rate.
    pub(crate) host_receive_bytes_per_second: Option<u64>,
}

impl TransferEstimate {
    pub(crate) fn wire_valid(&self) -> bool {
        self.bytes_per_second.is_none_or(|rate| rate > 0)
            && self.eta_seconds.is_none_or(|eta| eta <= MAX_ETA_SECONDS)
            && (self.eta_seconds.is_none() || self.bytes_per_second.is_some())
            && !(self.state == TransferState::Stalled && self.eta_seconds.is_some())
            && !(self.state == TransferState::Estimating && self.eta_seconds.is_some())
    }
}

#[derive(Debug, Default)]
pub(crate) struct TransferEstimator {
    samples: VecDeque<(Duration, u64)>,
    last_completed: Option<u64>,
    last_advance_at: Duration,
    first_advance_at: Option<Duration>,
    smoothed_rate: Option<f64>,
    last_rate_update: Option<Duration>,
    host_credited: u64,
    host_last: Option<(Duration, u64)>,
    host_rate: Option<f64>,
    stability_rates: VecDeque<f64>,
    last_stability_sample: Option<Duration>,
    eta_ready: bool,
    shown_eta: Option<(Duration, f64)>,
}

impl TransferEstimator {
    pub(crate) fn new() -> Self {
        Self::default()
    }

    /// Fold one observation into the estimate.
    ///
    /// `now` is monotonic time since the transfer began, `completed` and
    /// `total` are written logical bytes, and `host_received_total` is a
    /// cumulative host-wide receive counter when one is available.
    pub(crate) fn observe(
        &mut self,
        now: Duration,
        completed: u64,
        total: u64,
        host_received_total: Option<u64>,
    ) -> TransferEstimate {
        match self.last_completed {
            None => self.last_completed = Some(completed),
            Some(previous) if completed > previous => {
                self.last_completed = Some(completed);
                self.last_advance_at = now;
                self.first_advance_at.get_or_insert(now);
            }
            Some(_) => {}
        }
        self.record_sample(now, completed);
        self.observe_host(now, host_received_total);

        let idle = now.saturating_sub(self.last_advance_at);
        let idle_for_a_while = idle >= IDLE_AFTER;
        let warm = self
            .first_advance_at
            .is_some_and(|first| now.saturating_sub(first) >= WARMUP);

        // Freeze the rate while written bytes are idle: Xet keeps the network
        // busy during those gaps, so decaying toward zero would only recreate
        // the absurd ETA this estimator exists to prevent.
        if warm && !idle_for_a_while {
            if let Some(window_rate) = self.window_rate(now) {
                self.record_stability_sample(now, window_rate);
                self.smoothed_rate = Some(match (self.smoothed_rate, self.last_rate_update) {
                    (Some(rate), Some(updated)) => {
                        let dt = now.saturating_sub(updated).as_secs_f64();
                        let alpha = 1.0 - (-dt / RATE_TIME_CONSTANT.as_secs_f64()).exp();
                        rate + alpha * (window_rate - rate)
                    }
                    _ => window_rate,
                });
                self.last_rate_update = Some(now);
            }
        } else if self.smoothed_rate.is_some() {
            self.last_rate_update = Some(now);
        }

        let complete = total > 0 && completed >= total;
        let network_active = self
            .host_rate
            .is_some_and(|rate| rate >= NETWORK_ACTIVE_FLOOR);
        let rate = self
            .smoothed_rate
            .filter(|rate| rate.is_finite() && *rate >= 1.0);
        if !self.eta_ready && rate.is_some() && self.rate_is_stable(now, completed, total) {
            self.eta_ready = true;
        }
        let state = if complete {
            TransferState::Transferring
        } else if idle_for_a_while {
            if network_active {
                TransferState::Buffering
            } else {
                TransferState::Stalled
            }
        } else if !self.eta_ready {
            TransferState::Estimating
        } else {
            TransferState::Transferring
        };
        let eta_seconds = if complete {
            rate.map(|_| 0)
        } else {
            match state {
                TransferState::Estimating | TransferState::Stalled => None,
                TransferState::Transferring | TransferState::Buffering if self.eta_ready => rate
                    .map(|rate| {
                        let remaining = total.saturating_sub(completed) as f64;
                        self.bounded_eta(now, remaining / rate)
                    }),
                TransferState::Transferring | TransferState::Buffering => None,
            }
        };

        TransferEstimate {
            state,
            bytes_per_second: rate.map(|rate| rate.round().clamp(1.0, u64::MAX as f64) as u64),
            eta_seconds,
            idle_ms: idle.as_millis().min(u128::from(u64::MAX)) as u64,
            host_received_bytes: self.host_last.map(|_| self.host_credited),
            host_receive_bytes_per_second: self
                .host_rate
                .map(|rate| rate.round().clamp(0.0, u64::MAX as f64) as u64),
        }
    }

    fn record_stability_sample(&mut self, now: Duration, window_rate: f64) {
        let due = self
            .last_stability_sample
            .is_none_or(|at| now.saturating_sub(at) >= STABILITY_SPACING);
        if !due {
            return;
        }
        self.last_stability_sample = Some(now);
        self.stability_rates.push_back(window_rate);
        while self.stability_rates.len() > STABILITY_SAMPLES {
            self.stability_rates.pop_front();
        }
    }

    fn rate_is_stable(&self, now: Duration, completed: u64, total: u64) -> bool {
        let long_enough = now >= ETA_MIN_ELAPSED;
        let enough_bytes = total > 0 && completed as f64 >= total as f64 * ETA_MIN_FRACTION;
        let slow_but_long = now >= ETA_SLOW_ELAPSED;
        if !long_enough || !(enough_bytes || slow_but_long) {
            return false;
        }
        if self.stability_rates.len() < STABILITY_SAMPLES {
            return false;
        }
        let (min, max) = self
            .stability_rates
            .iter()
            .fold((f64::INFINITY, 0.0_f64), |(lo, hi), rate| {
                (lo.min(*rate), hi.max(*rate))
            });
        min > 0.0 && max / min <= STABILITY_TOLERANCE
    }

    /// Count the previously shown ETA down by elapsed time and limit how far
    /// the new estimate may move from it, then cap it.
    fn bounded_eta(&mut self, now: Duration, raw: f64) -> u64 {
        let raw = if raw.is_finite() {
            raw.max(0.0)
        } else {
            MAX_ETA_SECONDS as f64
        };
        let shown = match self.shown_eta {
            Some((at, previous)) => {
                let dt = now.saturating_sub(at).as_secs_f64();
                let expected = (previous - dt).max(1.0);
                let factor = 2_f64.powf(dt / ETA_DOUBLING_INTERVAL.as_secs_f64());
                raw.clamp(expected / factor, expected * factor)
            }
            None => raw,
        }
        .min(MAX_ETA_SECONDS as f64);
        self.shown_eta = Some((now, shown));
        shown.ceil() as u64
    }

    fn record_sample(&mut self, now: Duration, completed: u64) {
        let due = self
            .samples
            .back()
            .is_none_or(|(at, _)| now.saturating_sub(*at) >= SAMPLE_SPACING);
        if due {
            self.samples.push_back((now, completed));
        }
        // Keep exactly one anchor at or before the window start.
        let window_start = now.saturating_sub(RATE_WINDOW);
        while self.samples.len() > 2 && self.samples[1].0 <= window_start {
            self.samples.pop_front();
        }
    }

    fn window_rate(&self, now: Duration) -> Option<f64> {
        let (anchor_at, anchor_bytes) = *self.samples.front()?;
        let current = self.last_completed?;
        let span = now.saturating_sub(anchor_at);
        if span < SAMPLE_SPACING {
            return None;
        }
        Some(current.saturating_sub(anchor_bytes) as f64 / span.as_secs_f64())
    }

    fn observe_host(&mut self, now: Duration, total: Option<u64>) {
        let Some(total) = total else {
            return;
        };
        let Some((last_at, last_total)) = self.host_last else {
            self.host_last = Some((now, total));
            return;
        };
        if total < last_total {
            // Counter reset (interface removed or wrapped): keep the bytes
            // already credited to this transfer and restart from here.
            self.host_last = Some((now, total));
            return;
        }
        let dt = now.saturating_sub(last_at);
        if dt < SAMPLE_SPACING {
            return;
        }
        let delta = total - last_total;
        self.host_credited = self.host_credited.saturating_add(delta);
        let instant = delta as f64 / dt.as_secs_f64();
        let alpha = 1.0 - (-dt.as_secs_f64() / HOST_RATE_TIME_CONSTANT.as_secs_f64()).exp();
        self.host_rate = Some(match self.host_rate {
            Some(rate) => rate + alpha * (instant - rate),
            None => instant,
        });
        self.host_last = Some((now, total));
    }
}

/// Cumulative host-wide receive counter over non-loopback interfaces.
///
/// macOS exposes no supported per-process network counter, and `hf-hub` does
/// not forward Xet's own transfer bytes, so this is an honest proxy: it
/// includes unrelated traffic and is labeled "host network" everywhere it is
/// shown. It never feeds the percentage or ETA.
pub(crate) struct HostNetworkCounter {
    networks: sysinfo::Networks,
}

impl HostNetworkCounter {
    pub(crate) fn new() -> Self {
        Self {
            networks: sysinfo::Networks::new_with_refreshed_list(),
        }
    }

    pub(crate) fn total_received(&mut self) -> Option<u64> {
        self.networks.refresh();
        let mut seen = false;
        let total = self
            .networks
            .iter()
            .filter(|(name, _)| !name.starts_with("lo"))
            .map(|(_, data)| {
                seen = true;
                data.total_received()
            })
            .fold(0_u64, u64::saturating_add);
        seen.then_some(total)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const MIB: u64 = 1024 * 1024;
    const GIB: u64 = 1024 * MIB;

    fn secs(value: f64) -> Duration {
        Duration::from_secs_f64(value)
    }

    /// Replays the bursty written-byte pattern measured for issue #249
    /// (23.3 GiB `APEX-Q5_K_M.gguf`, relative seconds, written bytes) at 100 ms
    /// resolution with a steady 300 MiB/s host network underneath.
    fn observed_burst_schedule() -> Vec<(f64, u64)> {
        vec![
            (0.0, 0),
            (2.0, 202 * MIB),
            (61.0, 973 * MIB),
            (86.0, 2_458 * MIB),
            (98.0, 5 * GIB),
            (110.0, 9 * GIB),
            (134.0, 12 * GIB + 100 * MIB),
            (159.0, 13 * GIB + 300 * MIB),
            (183.0, 15 * GIB + 600 * MIB),
            (220.0, 16 * GIB + 600 * MIB),
            (232.0, 23 * GIB + 300 * MIB),
        ]
    }

    fn written_at(schedule: &[(f64, u64)], t: f64) -> u64 {
        schedule
            .iter()
            .rev()
            .find(|(at, _)| *at <= t)
            .map(|(_, bytes)| *bytes)
            .unwrap_or(0)
    }

    fn replay(
        schedule: &[(f64, u64)],
        total: u64,
        until: f64,
        host_rate: Option<u64>,
    ) -> Vec<(f64, TransferEstimate)> {
        let mut estimator = TransferEstimator::new();
        let mut out = Vec::new();
        let steps = (until * 10.0) as u64;
        for step in 0..=steps {
            let t = step as f64 / 10.0;
            let host = host_rate.map(|rate| 7 * GIB + (rate as f64 * t) as u64);
            let estimate = estimator.observe(secs(t), written_at(schedule, t), total, host);
            out.push((t, estimate));
        }
        out
    }

    #[test]
    fn bursty_xet_pattern_never_prints_absurd_or_swinging_eta() {
        let total = 23 * GIB + 300 * MIB;
        let schedule = observed_burst_schedule();
        let trace = replay(&schedule, total, 240.0, Some(300 * MIB));

        let etas: Vec<(f64, u64)> = trace
            .iter()
            .filter_map(|(t, estimate)| estimate.eta_seconds.map(|eta| (*t, eta)))
            .collect();
        assert!(!etas.is_empty());
        for (_, estimate) in &trace {
            assert!(estimate.wire_valid(), "{estimate:?}");
            if let Some(eta) = estimate.eta_seconds {
                assert!(eta <= MAX_ETA_SECONDS);
            }
        }
        // The raw Xet rate produced 8h 32m at t~25 s and >2,000 years
        // elsewhere. Once a rate exists, the smoothed ETA stays under an hour
        // for a transfer that actually finished in ~4 minutes.
        assert!(
            etas.iter().all(|(_, eta)| *eta < 3600),
            "ETA exceeded an hour: {:?}",
            etas.iter().max_by_key(|(_, eta)| *eta)
        );
        // After the transfer is under way (first 90 s), sampling every 10 s
        // the ETA never jumps by more than 3x between adjacent samples.
        let sampled: Vec<u64> = trace
            .iter()
            .filter(|(t, _)| *t >= 90.0 && *t < 232.0 && (*t * 10.0) as u64 % 100 == 0)
            .filter_map(|(_, estimate)| estimate.eta_seconds)
            .collect();
        assert!(sampled.len() >= 5, "{sampled:?}");
        for pair in sampled.windows(2) {
            let (low, high) = (pair[0].min(pair[1]).max(1), pair[0].max(pair[1]));
            assert!(high <= low * 3, "ETA swung {pair:?} in {sampled:?}");
        }
    }

    /// Field test of the first #249 fix (23.3 GiB APEX-Q5_K_M, empty cache,
    /// finished in 256 s): it printed "ETA 1h 29m" at 1% and "31m 27s" at
    /// ~114 s. Seconds since start -> displayed (written) bytes.
    fn field_test_schedule() -> Vec<(f64, u64)> {
        vec![
            (0.0, 0),
            (7.0, 138 * MIB),
            (14.0, 267 * MIB),
            (28.0, 1_638 * MIB),
            (114.0, 2_161 * MIB),
            (171.0, 4_915 * MIB),
            (228.0, 9_922 * MIB),
            (256.0, 23_880 * MIB),
        ]
    }

    #[test]
    fn field_test_sequence_withholds_eta_until_stable_and_never_overshoots() {
        let total = 23_880 * MIB;
        let finished_at = 256.0;
        let schedule = field_test_schedule();
        for host_rate in [None, Some(200 * MIB)] {
            let trace = replay(&schedule, total, 260.0, host_rate);
            for (t, estimate) in &trace {
                assert!(estimate.wire_valid(), "t={t} {estimate:?}");
                let written = written_at(&schedule, *t);
                let Some(eta) = estimate.eta_seconds else {
                    continue;
                };
                if *t >= finished_at {
                    continue;
                }
                assert!(
                    *t >= ETA_MIN_ELAPSED.as_secs_f64()
                        && written as f64 >= total as f64 * ETA_MIN_FRACTION,
                    "ETA {eta}s shown before the stability rule at t={t}"
                );
                let true_remaining = finished_at - t;
                assert!(
                    eta as f64 <= 3.0 * true_remaining.max(1.0),
                    "t={t}: ETA {eta}s vs true remaining {true_remaining}s"
                );
            }
            // The two values the field test complained about are gone.
            for t in [7.5, 114.5] {
                let (_, estimate) = trace.iter().find(|(at, _)| (*at - t).abs() < 0.05).unwrap();
                assert_eq!(estimate.eta_seconds, None, "t={t} {estimate:?}");
            }
            let (_, done) = trace.last().unwrap();
            assert_eq!(done.eta_seconds, Some(0));
        }
    }

    #[test]
    fn shown_eta_moves_by_a_bounded_factor_per_update() {
        let total = 20 * GIB;
        let mut estimator = TransferEstimator::new();
        let mut previous: Option<(f64, u64)> = None;
        for step in 0..=3000_u64 {
            let t = step as f64 / 10.0;
            // 100 MiB/s steady, then the source suddenly drops to 10 MiB/s.
            let written = if t <= 100.0 {
                100.0 * MIB as f64 * t
            } else {
                100.0 * MIB as f64 * 100.0 + 10.0 * MIB as f64 * (t - 100.0)
            } as u64;
            let estimate = estimator.observe(secs(t), written, total, None);
            if let Some(eta) = estimate.eta_seconds {
                if let Some((at, before)) = previous {
                    let dt = t - at;
                    let expected = (before as f64 - dt).max(1.0);
                    let factor = 2_f64.powf(dt / ETA_DOUBLING_INTERVAL.as_secs_f64());
                    assert!(
                        eta as f64 <= expected * factor + 1.0
                            && eta as f64 + 1.0 >= expected / factor,
                        "t={t}: {before}s -> {eta}s"
                    );
                }
                previous = Some((t, eta));
            }
        }
        assert!(previous.is_some());
    }

    #[test]
    fn flat_written_bytes_with_busy_network_reports_buffering_not_hung() {
        let total = 23 * GIB + 300 * MIB;
        let schedule = observed_burst_schedule();
        let trace = replay(&schedule, total, 60.0, Some(300 * MIB));

        let at = |t: f64| {
            trace
                .iter()
                .find(|(at, _)| (*at - t).abs() < 0.05)
                .map(|(_, estimate)| *estimate)
                .unwrap()
        };
        // 40 s into the first ~59 s flat period: written bytes idle, network busy.
        let estimate = at(42.0);
        assert_eq!(estimate.state, TransferState::Buffering, "{estimate:?}");
        assert!(estimate.idle_ms >= 39_000, "{estimate:?}");
        // The received counter keeps advancing every second during the gap.
        let received_40 = at(40.0).host_received_bytes.unwrap();
        let received_42 = at(42.0).host_received_bytes.unwrap();
        assert!(
            received_42 >= received_40 + 500 * MIB,
            "{received_40} {received_42}"
        );
        assert!(estimate.host_receive_bytes_per_second.unwrap() >= 250 * MIB);
    }

    #[test]
    fn genuine_stall_is_reported_as_stalled_without_eta() {
        let total = 10 * GIB;
        let mut estimator = TransferEstimator::new();
        let mut host_total = 0_u64;
        let mut last = TransferEstimate::default();
        for step in 0..=900_u64 {
            let t = step as f64 / 10.0;
            // 100 MiB/s for 60 s, then nothing at all (written or network).
            let moving = t <= 60.0;
            if moving {
                host_total = (100.0 * MIB as f64 * t) as u64;
            }
            let written = (100.0 * MIB as f64 * t.min(60.0)) as u64;
            last = estimator.observe(secs(t), written, total, Some(host_total));
            if (50.0..60.0).contains(&t) {
                assert_eq!(last.state, TransferState::Transferring, "t={t} {last:?}");
                assert!(last.eta_seconds.is_some());
            }
            if t > 60.0 && t < 74.9 {
                assert_ne!(last.state, TransferState::Stalled, "t={t} {last:?}");
            }
        }
        assert_eq!(last.state, TransferState::Stalled, "{last:?}");
        assert_eq!(last.eta_seconds, None);
        assert!(last.idle_ms >= 29_000);
        assert!(last.wire_valid());
    }

    #[test]
    fn no_host_counter_falls_back_to_stalled_after_idle() {
        let total = GIB;
        let mut estimator = TransferEstimator::new();
        let estimate = estimator.observe(secs(0.0), 0, total, None);
        assert_eq!(estimate.state, TransferState::Estimating);
        let estimate = estimator.observe(secs(16.0), 0, total, None);
        assert_eq!(estimate.state, TransferState::Stalled);
        assert_eq!(estimate.host_received_bytes, None);
        assert_eq!(estimate.eta_seconds, None);
    }

    #[test]
    fn estimating_until_rate_is_stable_then_steady_rate_is_accurate() {
        let total = 10 * GIB;
        let mut estimator = TransferEstimator::new();
        let rate = 50 * MIB;
        for step in 0..=1200_u64 {
            let t = step as f64 / 10.0;
            let estimate = estimator.observe(secs(t), (rate as f64 * t) as u64, total, None);
            if t < ETA_MIN_ELAPSED.as_secs_f64() {
                assert_eq!(estimate.state, TransferState::Estimating, "t={t}");
                assert_eq!(estimate.eta_seconds, None);
            }
            if t >= 100.0 {
                let reported = estimate.bytes_per_second.unwrap();
                assert!(reported.abs_diff(rate) < rate / 50, "t={t} {reported}");
                let expected_eta = (total - (rate as f64 * t) as u64) / rate;
                assert!(
                    estimate.eta_seconds.unwrap().abs_diff(expected_eta) <= 3,
                    "t={t} {estimate:?}"
                );
            }
        }
    }

    #[test]
    fn very_slow_transfer_eta_is_capped_not_years() {
        let total = 25 * GIB;
        let mut estimator = TransferEstimator::new();
        let mut estimate = TransferEstimate::default();
        for step in 0..=2400_u64 {
            let t = step as f64 / 10.0;
            // 1 KiB/s for a 25 GiB file would be ~830 years.
            estimate = estimator.observe(secs(t), (1024.0 * t) as u64, total, None);
        }
        assert_eq!(estimate.state, TransferState::Transferring);
        assert_eq!(estimate.eta_seconds, Some(MAX_ETA_SECONDS));
        assert!(estimate.wire_valid());
    }

    #[test]
    fn completed_transfer_reports_zero_eta() {
        let mut estimator = TransferEstimator::new();
        for step in 0..=200_u64 {
            let t = step as f64 / 10.0;
            estimator.observe(secs(t), step * MIB, 200 * MIB, None);
        }
        let estimate = estimator.observe(secs(20.1), 200 * MIB, 200 * MIB, None);
        assert_eq!(estimate.state, TransferState::Transferring);
        assert_eq!(estimate.eta_seconds, Some(0));
    }

    #[test]
    fn host_counter_reset_does_not_underflow_received_bytes() {
        let mut estimator = TransferEstimator::new();
        estimator.observe(secs(0.0), 0, GIB, Some(10 * GIB));
        let before = estimator.observe(secs(2.0), 0, GIB, Some(10 * GIB + 100 * MIB));
        assert_eq!(before.host_received_bytes, Some(100 * MIB));
        let after = estimator.observe(secs(3.0), 0, GIB, Some(5 * MIB));
        assert_eq!(after.host_received_bytes, Some(100 * MIB));
        let later = estimator.observe(secs(4.0), 0, GIB, Some(15 * MIB));
        assert_eq!(later.host_received_bytes, Some(110 * MIB));
    }

    #[test]
    fn wire_validation_rejects_contradictory_estimates() {
        assert!(TransferEstimate::default().wire_valid());
        assert!(!TransferEstimate {
            state: TransferState::Stalled,
            bytes_per_second: Some(1),
            eta_seconds: Some(1),
            ..TransferEstimate::default()
        }
        .wire_valid());
        assert!(!TransferEstimate {
            state: TransferState::Transferring,
            bytes_per_second: Some(1),
            eta_seconds: Some(MAX_ETA_SECONDS + 1),
            ..TransferEstimate::default()
        }
        .wire_valid());
        assert!(!TransferEstimate {
            state: TransferState::Transferring,
            bytes_per_second: Some(0),
            ..TransferEstimate::default()
        }
        .wire_valid());
    }
}
