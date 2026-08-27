//! Operator-facing progress for native DeepSeek-V4 requests.
//!
//! These events deliberately use `tracing::info!`: `hf2q serve` remains
//! quiet at its default warning filter, while `-v` makes long agentic
//! prefills and buffered tool-call decodes observable. Only token counts,
//! rates, cache decisions, and timing are logged. Prompt text, decoded text,
//! tool names, and tool arguments never enter this surface.

use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

use crate::serve::operator_ui;

const REPORT_INTERVAL: Duration = Duration::from_secs(5);
static NEXT_REQUEST_ID: AtomicU64 = AtomicU64::new(1);

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(super) struct LatencyGapReceipt {
    observations: usize,
    first: Option<Duration>,
    last: Option<Duration>,
    max_gap: Duration,
}

impl LatencyGapReceipt {
    pub(super) fn with_origin() -> Self {
        Self {
            last: Some(Duration::ZERO),
            ..Self::default()
        }
    }

    pub(super) fn observe(&mut self, at: Duration) -> anyhow::Result<()> {
        if let Some(previous) = self.last {
            anyhow::ensure!(
                at >= previous,
                "DeepSeek-V4 latency receipt moved backwards: previous={previous:?}, current={at:?}"
            );
            self.max_gap = self.max_gap.max(at - previous);
        }
        self.first.get_or_insert(at);
        self.last = Some(at);
        self.observations = self.observations.saturating_add(1);
        Ok(())
    }

    pub(super) fn observations(&self) -> usize {
        self.observations
    }

    pub(super) fn first(&self) -> Option<Duration> {
        self.first
    }

    pub(super) fn max_gap(&self) -> Duration {
        self.max_gap
    }

    #[cfg(test)]
    fn require_bound(&self, minimum_observations: usize, bound: Duration) -> anyhow::Result<()> {
        anyhow::ensure!(
            self.observations >= minimum_observations,
            "DeepSeek-V4 latency receipt is incomplete: {} observations, need {minimum_observations}",
            self.observations
        );
        anyhow::ensure!(
            self.max_gap <= bound,
            "DeepSeek-V4 latency gap {:?} exceeds {:?}",
            self.max_gap,
            bound
        );
        Ok(())
    }
}

pub(super) struct RequestProgress {
    id: u64,
    started: Instant,
    prompt_tokens: usize,
    max_tokens: usize,
    cached_tokens: usize,
    prefill_work_tokens: usize,
    prefill_completed_tokens: usize,
    last_prefill_report: Duration,
    last_prefill_report_tokens: usize,
    mixed_prefill_reported: bool,
    decode_started: Option<Instant>,
    last_decode_report: Duration,
    first_semantic_reported: bool,
}

impl RequestProgress {
    pub(super) fn start(mode: &'static str, prompt_tokens: usize, max_tokens: usize) -> Self {
        let id = NEXT_REQUEST_ID.fetch_add(1, Ordering::Relaxed);
        tracing::info!(
            request_id = id,
            mode,
            prompt_tokens,
            max_tokens,
            "DeepSeek-V4 request started"
        );
        operator_ui::request_started("deepseek4", id, None, mode, prompt_tokens, max_tokens);
        Self {
            id,
            started: Instant::now(),
            prompt_tokens,
            max_tokens,
            cached_tokens: 0,
            prefill_work_tokens: 0,
            prefill_completed_tokens: 0,
            last_prefill_report: Duration::ZERO,
            last_prefill_report_tokens: 0,
            mixed_prefill_reported: false,
            decode_started: None,
            last_decode_report: Duration::ZERO,
            first_semantic_reported: false,
        }
    }

    pub(super) fn id(&self) -> u64 {
        self.id
    }

    pub(super) fn plan_prefill(
        &mut self,
        cached_tokens: usize,
        work_tokens: usize,
        cache_action: &'static str,
    ) {
        self.cached_tokens = cached_tokens;
        self.prefill_work_tokens = work_tokens;
        self.prefill_completed_tokens = 0;
        self.last_prefill_report = self.started.elapsed();
        self.last_prefill_report_tokens = 0;
        tracing::info!(
            request_id = self.id,
            prompt_tokens = self.prompt_tokens,
            cached_tokens,
            suffix_tokens = self.prompt_tokens.saturating_sub(cached_tokens),
            work_tokens,
            cache = cache_action,
            "DeepSeek-V4 prefill planned"
        );
        operator_ui::prefill_planned("deepseek4", self.id, cached_tokens, work_tokens);
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn cache_reset_diagnostic(
        &self,
        live_tokens: usize,
        live_position: usize,
        live_common_prefix_tokens: usize,
        recovery_tokens: usize,
        recovery_position: usize,
        recovery_common_prefix_tokens: usize,
        cache_poisoned: bool,
        matrix_prefill_forced: bool,
        cache_grew: bool,
    ) {
        tracing::info!(
            request_id = self.id,
            prompt_tokens = self.prompt_tokens,
            live_tokens,
            live_position,
            live_common_prefix_tokens,
            recovery_tokens,
            recovery_position,
            recovery_common_prefix_tokens,
            cache_poisoned,
            matrix_prefill_forced,
            cache_grew,
            "DeepSeek-V4 prefix reuse rejected"
        );
    }

    pub(super) fn advance_prefill(&mut self, tokens: usize) {
        self.prefill_completed_tokens = self
            .prefill_completed_tokens
            .saturating_add(tokens)
            .min(self.prefill_work_tokens);
        let elapsed = self.started.elapsed();
        let rate = tokens_per_second(self.prefill_completed_tokens, elapsed);
        operator_ui::prefill_progress(
            "deepseek4",
            self.id,
            self.prefill_completed_tokens,
            self.prefill_work_tokens,
            rate,
        );
        if !should_report(
            self.last_prefill_report,
            elapsed,
            self.prefill_completed_tokens,
            self.prefill_work_tokens,
        ) {
            return;
        }
        let interval_tokens = self
            .prefill_completed_tokens
            .saturating_sub(self.last_prefill_report_tokens);
        let interval_elapsed = elapsed.saturating_sub(self.last_prefill_report);
        let interval_rate = tokens_per_second(interval_tokens, interval_elapsed);
        let percent = if self.prefill_work_tokens == 0 {
            100.0
        } else {
            100.0 * self.prefill_completed_tokens as f64 / self.prefill_work_tokens as f64
        };
        tracing::info!(
            request_id = self.id,
            processed_tokens = self.prefill_completed_tokens,
            work_tokens = self.prefill_work_tokens,
            percent,
            tokens_per_second = rate,
            interval_tokens,
            interval_seconds = interval_elapsed.as_secs_f64(),
            interval_tokens_per_second = interval_rate,
            elapsed_seconds = elapsed.as_secs_f64(),
            "DeepSeek-V4 prefill progress"
        );
        self.last_prefill_report = elapsed;
        self.last_prefill_report_tokens = self.prefill_completed_tokens;
    }

    /// Seal the first scheduler-capped prefill transaction for a genuinely
    /// mixed decode/prefill turn. The ordinary five-second progress record is
    /// cumulative and may include earlier solo work, so it cannot prove the
    /// peer-visible transaction boundary by itself.
    pub(super) fn mixed_prefill_slice(&mut self, chunk_tokens: usize, window_cap: usize) {
        if self.mixed_prefill_reported {
            return;
        }
        self.mixed_prefill_reported = true;
        tracing::info!(
            request_id = self.id,
            chunk_tokens,
            window_cap,
            "DeepSeek-V4 mixed prefill slice"
        );
    }

    /// Emit one boundary event so cancellation gates can prove they aborted
    /// after the request-local candidate checkpoint existed, rather than only
    /// during the early reusable suffix.
    pub(super) fn recovery_anchor_captured(&self, anchor_tokens: usize) {
        tracing::info!(
            request_id = self.id,
            anchor_tokens,
            "DeepSeek-V4 request recovery anchor captured"
        );
    }

    pub(super) fn finish_prefill(&mut self, duration: Duration) {
        let suffix_tokens = self.prompt_tokens.saturating_sub(self.cached_tokens);
        let rate = tokens_per_second(suffix_tokens, duration);
        tracing::info!(
            request_id = self.id,
            prompt_tokens = self.prompt_tokens,
            cached_tokens = self.cached_tokens,
            suffix_tokens,
            elapsed_seconds = duration.as_secs_f64(),
            tokens_per_second = rate,
            "DeepSeek-V4 prefill complete"
        );
    }

    pub(super) fn start_decode(&mut self) {
        self.decode_started = Some(Instant::now());
        self.last_decode_report = Duration::ZERO;
        tracing::info!(
            request_id = self.id,
            max_tokens = self.max_tokens,
            "DeepSeek-V4 decode started"
        );
        operator_ui::decode_progress("deepseek4", self.id, 0, self.max_tokens, 0.0);
    }

    pub(super) fn advance_decode(&mut self, generated_tokens: usize) {
        let Some(started) = self.decode_started else {
            return;
        };
        let elapsed = started.elapsed();
        let rate = tokens_per_second(generated_tokens, elapsed);
        operator_ui::decode_progress(
            "deepseek4",
            self.id,
            generated_tokens,
            self.max_tokens,
            rate,
        );
        if !should_report(
            self.last_decode_report,
            elapsed,
            generated_tokens,
            self.max_tokens,
        ) {
            return;
        }
        self.last_decode_report = elapsed;
        tracing::info!(
            request_id = self.id,
            generated_tokens,
            max_tokens = self.max_tokens,
            tokens_per_second = rate,
            elapsed_seconds = elapsed.as_secs_f64(),
            "DeepSeek-V4 decode progress"
        );
    }

    pub(super) fn first_semantic_token(&mut self, ttft: Duration) {
        if self.first_semantic_reported {
            return;
        }
        self.first_semantic_reported = true;
        tracing::info!(
            request_id = self.id,
            ttft_seconds = ttft.as_secs_f64(),
            "DeepSeek-V4 first semantic stream event"
        );
    }

    pub(super) fn mixed_latency_receipt(
        &self,
        scheduler_decode: LatencyGapReceipt,
        semantic_sse: LatencyGapReceipt,
    ) {
        tracing::info!(
            request_id = self.id,
            scheduler_decode_visits = scheduler_decode.observations(),
            scheduler_decode_first_visit_ms = scheduler_decode
                .first()
                .map(|value| value.as_secs_f64() * 1000.0),
            scheduler_decode_max_gap_ms = scheduler_decode.max_gap().as_secs_f64() * 1000.0,
            semantic_sse_events = semantic_sse.observations(),
            semantic_sse_first_event_ms = semantic_sse
                .first()
                .map(|value| value.as_secs_f64() * 1000.0),
            semantic_sse_max_gap_ms = semantic_sse.max_gap().as_secs_f64() * 1000.0,
            "DeepSeek-V4 slot latency receipt"
        );
    }

    pub(super) fn complete(
        &mut self,
        decoder_stop_reason: &'static str,
        completion_tokens: usize,
        semantic_ttft: Option<Duration>,
    ) {
        if let Some(ttft) = semantic_ttft {
            self.first_semantic_token(ttft);
        }
        let total = self.started.elapsed();
        let decode_rate = self.decode_started.map_or(0.0, |started| {
            completion_tokens as f64 / started.elapsed().as_secs_f64().max(f64::EPSILON)
        });
        tracing::info!(
            request_id = self.id,
            decoder_stop_reason,
            prompt_tokens = self.prompt_tokens,
            cached_tokens = self.cached_tokens,
            completion_tokens,
            total_seconds = total.as_secs_f64(),
            decode_tokens_per_second = decode_rate,
            "DeepSeek-V4 request complete"
        );
        operator_ui::request_finished("deepseek4", self.id, "complete");
    }

    pub(super) fn cancelled(&self) {
        tracing::info!(
            request_id = self.id,
            elapsed_seconds = self.started.elapsed().as_secs_f64(),
            "DeepSeek-V4 request cancelled"
        );
        operator_ui::request_finished("deepseek4", self.id, "cancelled");
    }

    pub(super) fn failed(&self, error: &anyhow::Error) {
        let error_chain = format!("{error:#}");
        tracing::error!(
            request_id = self.id,
            elapsed_seconds = self.started.elapsed().as_secs_f64(),
            error = %error_chain,
            "DeepSeek-V4 request failed"
        );
        operator_ui::request_finished("deepseek4", self.id, "failed");
    }
}

fn tokens_per_second(tokens: usize, duration: Duration) -> f64 {
    tokens as f64 / duration.as_secs_f64().max(f64::EPSILON)
}

fn should_report(previous: Duration, now: Duration, completed: usize, total: usize) -> bool {
    total > 0 && (completed >= total || now.saturating_sub(previous) >= REPORT_INTERVAL)
}

#[cfg(test)]
mod tests {
    use super::{should_report, tokens_per_second, LatencyGapReceipt, REPORT_INTERVAL};
    use std::time::Duration;

    #[test]
    fn interval_rate_uses_only_the_latest_progress_window() {
        assert_eq!(tokens_per_second(2_048, Duration::from_secs(8)), 256.0);
        assert_eq!(tokens_per_second(4_096, Duration::from_secs(8)), 512.0);
    }

    #[test]
    fn progress_is_throttled_until_interval_or_completion() {
        assert!(!should_report(
            Duration::ZERO,
            REPORT_INTERVAL - Duration::from_millis(1),
            50,
            100
        ));
        assert!(should_report(Duration::ZERO, REPORT_INTERVAL, 50, 100));
        assert!(should_report(
            Duration::from_secs(4),
            Duration::from_secs(4),
            100,
            100
        ));
    }

    #[test]
    fn empty_prefill_never_emits_a_progress_heartbeat() {
        assert!(!should_report(
            Duration::ZERO,
            Duration::from_secs(30),
            0,
            0
        ));
    }

    #[test]
    fn latency_gap_receipt_is_measurable_and_fails_closed() {
        let mut receipt = LatencyGapReceipt::with_origin();
        for at in [4, 11, 18, 26] {
            receipt.observe(Duration::from_millis(at)).unwrap();
        }
        assert_eq!(receipt.observations(), 4);
        assert_eq!(receipt.first(), Some(Duration::from_millis(4)));
        assert_eq!(receipt.max_gap(), Duration::from_millis(8));
        receipt.require_bound(4, Duration::from_millis(8)).unwrap();
        assert!(receipt.require_bound(5, Duration::from_millis(8)).is_err());
        assert!(receipt.require_bound(4, Duration::from_millis(7)).is_err());
        assert!(receipt.observe(Duration::from_millis(25)).is_err());
    }
}
