//! Operator-facing progress for native DeepSeek-V4 requests.
//!
//! These events deliberately use `tracing::info!`: `hf2q serve` remains
//! quiet at its default warning filter, while `-v` makes long agentic
//! prefills and buffered tool-call decodes observable. Only token counts,
//! rates, cache decisions, and timing are logged. Prompt text, decoded text,
//! tool names, and tool arguments never enter this surface.

use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

const REPORT_INTERVAL: Duration = Duration::from_secs(5);
static NEXT_REQUEST_ID: AtomicU64 = AtomicU64::new(1);

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
            decode_started: None,
            last_decode_report: Duration::ZERO,
            first_semantic_reported: false,
        }
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
        let rate = tokens_per_second(self.prefill_completed_tokens, elapsed);
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
    }

    pub(super) fn advance_decode(&mut self, generated_tokens: usize) {
        let Some(started) = self.decode_started else {
            return;
        };
        let elapsed = started.elapsed();
        if !should_report(
            self.last_decode_report,
            elapsed,
            generated_tokens,
            self.max_tokens,
        ) {
            return;
        }
        self.last_decode_report = elapsed;
        let rate = tokens_per_second(generated_tokens, elapsed);
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
    }

    pub(super) fn cancelled(&self) {
        tracing::info!(
            request_id = self.id,
            elapsed_seconds = self.started.elapsed().as_secs_f64(),
            "DeepSeek-V4 request cancelled"
        );
    }

    pub(super) fn failed(&self, error: &anyhow::Error) {
        let error_chain = format!("{error:#}");
        tracing::error!(
            request_id = self.id,
            elapsed_seconds = self.started.elapsed().as_secs_f64(),
            error = %error_chain,
            "DeepSeek-V4 request failed"
        );
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
    use super::{should_report, tokens_per_second, REPORT_INTERVAL};
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
}
