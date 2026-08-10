//! Foreground serve dashboard and tracing handoff.
//!
//! The inference worker only performs `try_send` calls into this module. A
//! slow terminal can therefore never delay cache, scheduler, or GPU work.

use std::collections::{BTreeMap, VecDeque};
use std::io::{self, IsTerminal, Write};
use std::sync::mpsc::{self, Receiver, SyncSender};
use std::sync::Mutex;
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use tracing_subscriber::fmt::MakeWriter;

use crate::cli::OperatorUiArg;

mod render;
use render::{draw, enter_terminal, restore_terminal};

const EVENT_CAPACITY: usize = 256;
const REDRAW_INTERVAL: Duration = Duration::from_millis(200);
const FINISHED_RETENTION: Duration = Duration::from_secs(4);
const MAX_LOG_LINES: usize = 4;
const STRUCTURED_PROGRESS_MESSAGES: &[&str] = &[
    "chat completion request received",
    "chat completion prepared; dispatching",
    "streaming request submitted",
    "streaming request enqueued",
    "bounded prefill chunk complete",
    "request started",
    "prefill planned",
    "prefill progress",
    "decode started",
    "decode progress",
];

static EVENT_SINK: Mutex<Option<SyncSender<Event>>> = Mutex::new(None);

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
struct RequestKey {
    family: &'static str,
    id: u64,
}

#[derive(Debug)]
enum Event {
    Started {
        key: RequestKey,
        slot: Option<u32>,
        mode: &'static str,
        prompt_tokens: usize,
        max_tokens: usize,
    },
    PrefillPlanned {
        key: RequestKey,
        cached_tokens: usize,
        work_tokens: usize,
    },
    PrefillProgress {
        key: RequestKey,
        processed_tokens: usize,
        work_tokens: usize,
        tokens_per_second: f64,
    },
    DecodeProgress {
        key: RequestKey,
        generated_tokens: usize,
        max_tokens: usize,
        tokens_per_second: f64,
    },
    Finished {
        key: RequestKey,
        outcome: &'static str,
    },
    Unhealthy(String),
    Log(String),
    Stop,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Phase {
    Queued,
    Prefill,
    Decode,
    Complete,
    Cancelled,
    Failed,
}

struct RequestView {
    key: RequestKey,
    slot: Option<u32>,
    mode: &'static str,
    phase: Phase,
    prompt_tokens: usize,
    max_tokens: usize,
    cached_tokens: usize,
    completed_tokens: usize,
    work_tokens: usize,
    generated_tokens: usize,
    rate: f64,
    started: Instant,
    finished_at: Option<Instant>,
}

struct DashboardState {
    model: String,
    family: String,
    endpoint: String,
    max_slots: u32,
    started: Instant,
    requests: BTreeMap<RequestKey, RequestView>,
    logs: VecDeque<String>,
    unhealthy: Option<String>,
}

pub(crate) struct OperatorUiGuard {
    sender: Option<SyncSender<Event>>,
    thread: Option<JoinHandle<()>>,
}

impl Drop for OperatorUiGuard {
    fn drop(&mut self) {
        if let Ok(mut sink) = EVENT_SINK.lock() {
            *sink = None;
        }
        if let Some(sender) = self.sender.take() {
            // Never let a saturated presentation queue delay server shutdown.
            // Dropping the last sender also terminates the renderer after it
            // drains any events already in flight when Stop cannot be queued.
            let _ = sender.try_send(Event::Stop);
        }
        if let Some(thread) = self.thread.take() {
            let _ = thread.join();
        }
    }
}

/// Restores the caller's terminal exactly once even if the renderer unwinds.
struct TerminalGuard;

impl Drop for TerminalGuard {
    fn drop(&mut self) {
        restore_terminal();
    }
}

/// A per-tracing-event buffer. Before the dashboard starts it preserves the
/// normal stderr stream; while active it forwards whole lines to the bounded
/// dashboard channel instead of painting through the live screen.
pub(crate) struct LogLineWriter {
    bytes: Vec<u8>,
}

impl Write for LogLineWriter {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        self.bytes.extend_from_slice(buf);
        Ok(buf.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

impl Drop for LogLineWriter {
    fn drop(&mut self) {
        if self.bytes.is_empty() {
            return;
        }
        let line = String::from_utf8_lossy(&self.bytes).into_owned();
        if publish(Event::Log(line)) {
            return;
        }
        let mut stderr = io::stderr().lock();
        let _ = stderr.write_all(&self.bytes);
        let _ = stderr.flush();
    }
}

#[derive(Clone, Copy)]
pub(crate) struct LogMakeWriter;

impl<'a> MakeWriter<'a> for LogMakeWriter {
    type Writer = LogLineWriter;

    fn make_writer(&'a self) -> Self::Writer {
        LogLineWriter { bytes: Vec::new() }
    }
}

pub(crate) fn start(
    mode: OperatorUiArg,
    text_logs: bool,
    model: String,
    family: String,
    endpoint: String,
    max_slots: u32,
) -> Result<Option<OperatorUiGuard>> {
    let active = activation_for_current_process(mode, text_logs)?;
    if !active {
        return Ok(None);
    }

    let (sender, receiver) = mpsc::sync_channel(EVENT_CAPACITY);
    let thread = std::thread::Builder::new()
        .name("hf2q-operator-ui".into())
        .spawn(move || render_loop(receiver, model, family, endpoint, max_slots))
        .context("spawn hf2q operator dashboard")?;
    let mut sink = EVENT_SINK
        .lock()
        .map_err(|_| anyhow::anyhow!("operator dashboard event sink poisoned"))?;
    anyhow::ensure!(sink.is_none(), "operator dashboard already active");
    *sink = Some(sender.clone());
    Ok(Some(OperatorUiGuard {
        sender: Some(sender),
        thread: Some(thread),
    }))
}

pub(crate) fn validate_mode(mode: OperatorUiArg, text_logs: bool) -> Result<()> {
    let _ = activation_for_current_process(mode, text_logs)?;
    Ok(())
}

fn activation_for_current_process(mode: OperatorUiArg, text_logs: bool) -> Result<bool> {
    resolve_activation(
        mode,
        text_logs,
        io::stderr().is_terminal(),
        std::env::var_os("CI").is_some(),
        std::env::var("TERM").ok().as_deref(),
    )
}

fn resolve_activation(
    mode: OperatorUiArg,
    text_logs: bool,
    stderr_tty: bool,
    ci: bool,
    term: Option<&str>,
) -> Result<bool> {
    match mode {
        OperatorUiArg::Plain => Ok(false),
        OperatorUiArg::Auto => Ok(text_logs && stderr_tty && !ci && term != Some("dumb")),
        OperatorUiArg::Dashboard => {
            anyhow::ensure!(
                text_logs,
                "--operator-ui dashboard cannot be combined with --log-format json"
            );
            anyhow::ensure!(
                stderr_tty,
                "--operator-ui dashboard requires an interactive stderr terminal"
            );
            Ok(true)
        }
    }
}

pub(crate) fn request_started(
    family: &'static str,
    id: u64,
    slot: Option<u32>,
    mode: &'static str,
    prompt_tokens: usize,
    max_tokens: usize,
) {
    let _ = publish(Event::Started {
        key: RequestKey { family, id },
        slot,
        mode,
        prompt_tokens,
        max_tokens,
    });
}

pub(crate) fn prefill_planned(
    family: &'static str,
    id: u64,
    cached_tokens: usize,
    work_tokens: usize,
) {
    let _ = publish(Event::PrefillPlanned {
        key: RequestKey { family, id },
        cached_tokens,
        work_tokens,
    });
}

pub(crate) fn prefill_progress(
    family: &'static str,
    id: u64,
    processed_tokens: usize,
    work_tokens: usize,
    tokens_per_second: f64,
) {
    let _ = publish(Event::PrefillProgress {
        key: RequestKey { family, id },
        processed_tokens,
        work_tokens,
        tokens_per_second,
    });
}

pub(crate) fn decode_progress(
    family: &'static str,
    id: u64,
    generated_tokens: usize,
    max_tokens: usize,
    tokens_per_second: f64,
) {
    let _ = publish(Event::DecodeProgress {
        key: RequestKey { family, id },
        generated_tokens,
        max_tokens,
        tokens_per_second,
    });
}

pub(crate) fn request_finished(family: &'static str, id: u64, outcome: &'static str) {
    let _ = publish(Event::Finished {
        key: RequestKey { family, id },
        outcome,
    });
}

pub(crate) fn engine_unhealthy(reason: String) {
    let _ = publish(Event::Unhealthy(reason));
}

/// Returns true whenever a dashboard owns the log line, including when its
/// bounded queue is full and the line is deliberately dropped.
fn publish(event: Event) -> bool {
    let Ok(mut sink) = EVENT_SINK.lock() else {
        return false;
    };
    let Some(sender) = sink.as_ref() else {
        return false;
    };
    match sender.try_send(event) {
        Ok(()) | Err(mpsc::TrySendError::Full(_)) => true,
        Err(mpsc::TrySendError::Disconnected(_)) => {
            *sink = None;
            false
        }
    }
}

fn render_loop(
    receiver: Receiver<Event>,
    model: String,
    family: String,
    endpoint: String,
    max_slots: u32,
) {
    let mut state = DashboardState {
        model,
        family,
        endpoint,
        max_slots,
        started: Instant::now(),
        requests: BTreeMap::new(),
        logs: VecDeque::new(),
        unhealthy: None,
    };
    enter_terminal();
    let _terminal = TerminalGuard;
    let mut running = true;
    while running {
        match receiver.recv_timeout(REDRAW_INTERVAL) {
            Ok(event) => running = apply_event(&mut state, event),
            Err(mpsc::RecvTimeoutError::Timeout) => {}
            Err(mpsc::RecvTimeoutError::Disconnected) => break,
        }
        while let Ok(event) = receiver.try_recv() {
            if !apply_event(&mut state, event) {
                running = false;
                break;
            }
        }
        let now = Instant::now();
        state.requests.retain(|_, request| {
            request
                .finished_at
                .is_none_or(|finished| now.duration_since(finished) < FINISHED_RETENTION)
        });
        draw(&state);
    }
}

fn apply_event(state: &mut DashboardState, event: Event) -> bool {
    match event {
        Event::Started {
            key,
            slot,
            mode,
            prompt_tokens,
            max_tokens,
        } => {
            state.requests.insert(
                key,
                RequestView {
                    key,
                    slot,
                    mode,
                    phase: Phase::Queued,
                    prompt_tokens,
                    max_tokens,
                    cached_tokens: 0,
                    completed_tokens: 0,
                    work_tokens: prompt_tokens,
                    generated_tokens: 0,
                    rate: 0.0,
                    started: Instant::now(),
                    finished_at: None,
                },
            );
        }
        Event::PrefillPlanned {
            key,
            cached_tokens,
            work_tokens,
        } => {
            if let Some(request) = state.requests.get_mut(&key) {
                request.phase = Phase::Prefill;
                request.cached_tokens = cached_tokens;
                request.work_tokens = work_tokens;
                request.completed_tokens = 0;
            }
        }
        Event::PrefillProgress {
            key,
            processed_tokens,
            work_tokens,
            tokens_per_second,
        } => {
            if let Some(request) = state.requests.get_mut(&key) {
                request.phase = Phase::Prefill;
                request.completed_tokens = processed_tokens;
                request.work_tokens = work_tokens;
                request.rate = tokens_per_second;
            }
        }
        Event::DecodeProgress {
            key,
            generated_tokens,
            max_tokens,
            tokens_per_second,
        } => {
            if let Some(request) = state.requests.get_mut(&key) {
                request.phase = Phase::Decode;
                request.generated_tokens = generated_tokens;
                request.max_tokens = max_tokens;
                request.rate = tokens_per_second;
            }
        }
        Event::Finished { key, outcome } => {
            if let Some(request) = state.requests.get_mut(&key) {
                request.phase = match outcome {
                    "cancelled" => Phase::Cancelled,
                    "failed" => Phase::Failed,
                    _ => Phase::Complete,
                };
                request.finished_at = Some(Instant::now());
            }
        }
        Event::Unhealthy(reason) => state.unhealthy = Some(reason),
        Event::Log(line) => {
            let line = console::strip_ansi_codes(&line).trim().to_owned();
            if !line.is_empty() && operator_log_is_not_request_progress(&line) {
                state.logs.push_back(line);
                while state.logs.len() > MAX_LOG_LINES {
                    state.logs.pop_front();
                }
            }
        }
        Event::Stop => return false,
    }
    true
}

fn operator_log_is_not_request_progress(line: &str) -> bool {
    !STRUCTURED_PROGRESS_MESSAGES
        .iter()
        .any(|message| line.contains(message))
}

#[cfg(test)]
mod tests {
    use super::render::{format_duration, progress_bar, render_request};
    use super::*;

    #[test]
    fn progress_bar_is_bounded() {
        assert_eq!(progress_bar(-1.0, 4), "░░░░");
        assert_eq!(progress_bar(50.0, 4), "██░░");
        assert_eq!(progress_bar(101.0, 4), "████");
    }

    #[test]
    fn duration_is_compact() {
        assert_eq!(format_duration(Duration::from_secs(9)), "9s");
        assert_eq!(format_duration(Duration::from_secs(69)), "1m09s");
    }

    #[test]
    fn activation_preserves_plain_machine_output_and_fails_closed_when_forced() {
        assert!(
            !resolve_activation(OperatorUiArg::Auto, true, false, false, Some("xterm"))
                .expect("auto pipe")
        );
        assert!(
            !resolve_activation(OperatorUiArg::Auto, false, true, false, Some("xterm"))
                .expect("auto JSON")
        );
        assert!(
            !resolve_activation(OperatorUiArg::Auto, true, true, true, Some("xterm"))
                .expect("auto CI")
        );
        assert!(
            !resolve_activation(OperatorUiArg::Plain, true, true, false, Some("xterm"))
                .expect("plain")
        );
        assert!(
            resolve_activation(OperatorUiArg::Dashboard, true, true, false, Some("xterm"))
                .expect("forced dashboard")
        );
        assert!(
            resolve_activation(OperatorUiArg::Dashboard, false, true, false, Some("xterm"))
                .is_err()
        );
        assert!(
            resolve_activation(OperatorUiArg::Dashboard, true, false, false, Some("xterm"))
                .is_err()
        );
    }

    #[test]
    fn recent_events_suppress_progress_that_has_a_structured_row() {
        assert!(!operator_log_is_not_request_progress(
            "INFO hf2q: Qwen35 bounded prefill chunk complete slot=1"
        ));
        assert!(!operator_log_is_not_request_progress(
            "INFO hf2q: DeepSeek-V4 decode progress request_id=2"
        ));
        assert!(operator_log_is_not_request_progress(
            "WARN hf2q: request rejected by physical KV budget"
        ));
    }

    #[test]
    fn state_tracks_cached_suffix_progress_without_log_lines() {
        let mut state = DashboardState {
            model: "test".into(),
            family: "qwen35".into(),
            endpoint: "http://127.0.0.1:8081".into(),
            max_slots: 4,
            started: Instant::now(),
            requests: BTreeMap::new(),
            logs: VecDeque::new(),
            unhealthy: None,
        };
        let key = RequestKey {
            family: "qwen35",
            id: 7,
        };
        assert!(apply_event(
            &mut state,
            Event::Started {
                key,
                slot: Some(1),
                mode: "slot-stream",
                prompt_tokens: 99_029,
                max_tokens: 8_192,
            }
        ));
        assert!(apply_event(
            &mut state,
            Event::PrefillPlanned {
                key,
                cached_tokens: 99_007,
                work_tokens: 22,
            }
        ));
        assert!(apply_event(
            &mut state,
            Event::PrefillProgress {
                key,
                processed_tokens: 11,
                work_tokens: 22,
                tokens_per_second: 110.0,
            }
        ));

        let request = state.requests.get(&key).expect("request row");
        assert_eq!(request.phase, Phase::Prefill);
        assert_eq!(request.cached_tokens, 99_007);
        assert_eq!(request.completed_tokens, 11);
        assert_eq!(request.work_tokens, 22);
        let row = render_request(request, 160);
        assert!(row.contains("50.0%"));
        assert!(row.contains("cache   99007"));
        assert!(state.logs.is_empty());

        assert!(apply_event(
            &mut state,
            Event::Unhealthy("engine_unhealthy: restart required".into())
        ));
        assert_eq!(
            state.unhealthy.as_deref(),
            Some("engine_unhealthy: restart required")
        );
    }
}
