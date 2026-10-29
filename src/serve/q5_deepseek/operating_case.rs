//! One selected process, immutable case bounds. No scheduler/model substitution.
use super::{operating_os, reviewed_budget::Policy};
use crate::serve::api::engine::{EngineMode, SamplingParams};
use anyhow::{ensure, Context, Result};
use serde::{Deserialize, Serialize};
use std::{
    cell::Cell,
    sync::{
        atomic::{AtomicBool, AtomicU64, Ordering},
        Mutex, OnceLock,
    },
    time::Instant,
};
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct Limits {
    pub queue_capacity: usize,
    pub max_enqueue_attempts: u64,
    pub max_prompt_tokens: usize,
    pub max_output_tokens: usize,
}
#[derive(Clone, Copy)]
struct Serve {
    queue: usize,
    slots: Option<u32>,
}
thread_local! {static SERVE:Cell<Option<Serve>>=const{Cell::new(None)};}
pub(crate) struct ServeScope {
    prior: Option<Serve>,
    _thread: std::marker::PhantomData<std::rc::Rc<()>>,
}
impl Drop for ServeScope {
    fn drop(&mut self) {
        SERVE.with(|x| x.set(self.prior));
    }
}
pub(crate) fn enter_serve(mode: EngineMode, queue: usize) -> Result<ServeScope> {
    let prior = SERVE.with(Cell::get);
    if super::selected()?.is_none() {
        return Ok(ServeScope {
            prior,
            _thread: std::marker::PhantomData,
        });
    }
    ensure!(prior.is_none(), "Q5 nested selected model load");
    SERVE.with(|x| {
        x.set(Some(Serve {
            queue,
            slots: match mode {
                EngineMode::SlotAware { max_slots } => Some(max_slots),
                _ => None,
            },
        }))
    });
    Ok(ServeScope {
        prior,
        _thread: std::marker::PhantomData,
    })
}
pub(super) fn require_actual(l: Limits) -> Result<()> {
    let s = SERVE
        .with(Cell::get)
        .context("Q5 selected budget requires actual serve configuration; direct loader refused")?;
    check_serve(l, s.queue, s.slots)
}
fn check_serve(l: Limits, queue: usize, slots: Option<u32>) -> Result<()> {
    ensure!(
        queue == l.queue_capacity && slots == Some(4),
        "Q5 actual queue/slots differ from reviewed case"
    );
    Ok(())
}

struct Runtime {
    ingress: std::sync::Arc<tokio::sync::Semaphore>,
    limits: Limits,
    policy: Policy,
    started: Instant,
    swap: u64,
    minimum: crate::inference::models::deepseek4::selected_budget_plan::Minimum,
    initial_os: operating_os::Sample,
    last: Mutex<Instant>,
    attempts: AtomicU64,
    failed: AtomicBool,
}
static ACTIVE: OnceLock<Runtime> = OnceLock::new();
pub(super) fn install(
    limits: Limits,
    policy: Policy,
    swap: u64,
    minimum: crate::inference::models::deepseek4::selected_budget_plan::Minimum,
    initial_os: operating_os::Sample,
) -> Result<()> {
    ACTIVE
        .set(Runtime {
            ingress: std::sync::Arc::new(tokio::sync::Semaphore::new(
                policy.max_http_inflight as usize,
            )),
            limits,
            policy,
            started: Instant::now(),
            swap,
            minimum,
            initial_os,
            last: Mutex::new(Instant::now()),
            attempts: AtomicU64::new(0),
            failed: AtomicBool::new(false),
        })
        .map_err(|_| anyhow::anyhow!("Q5 process already has a selected case; no retry"))
}
fn checked(f: impl FnOnce(&Runtime) -> Result<()>) -> Result<()> {
    let Some(r) = ACTIVE.get() else { return Ok(()) };
    ensure!(
        !r.failed.load(Ordering::Acquire),
        "Q5 sticky operating case failure"
    );
    let result = f(r);
    if result.is_err() {
        r.failed.store(true, Ordering::Release);
        return result;
    }
    ensure!(
        !r.failed.load(Ordering::Acquire),
        "Q5 concurrent sticky operating failure"
    );
    result
}
pub(crate) fn health() -> Result<()> {
    checked(|r| {
        ensure!(
            r.started.elapsed().as_millis() <= u128::from(r.policy.deadline_ms),
            "Q5 case deadline"
        );
        let mut last = r
            .last
            .lock()
            .map_err(|_| anyhow::anyhow!("Q5 OS observer poisoned"))?;
        if last.elapsed().as_millis() >= u128::from(r.policy.sample_interval_ms) {
            let begin = Instant::now();
            let s = operating_os::sample()?;
            ensure!(
                s.free >= r.policy.runtime_free_bytes
                    && s.swap <= r.swap
                    && s.page == r.policy.page_bytes
                    && s.rss <= r.policy.process_stop_bytes
                    && s.footprint <= r.policy.process_stop_bytes,
                "Q5 runtime Free/swap/page guard"
            );
            ensure!(
                begin.elapsed().as_millis() <= u128::from(r.policy.sample_interval_ms),
                "Q5 OS observation work exceeds declared reaction cadence"
            );
            *last = Instant::now();
        }
        Ok(())
    })
}
pub(crate) fn spawn(queue: usize, mode: EngineMode) -> Result<()> {
    checked(|r| {
        check_serve(
            r.limits,
            queue,
            match mode {
                EngineMode::SlotAware { max_slots } => Some(max_slots),
                _ => None,
            },
        )
    })
}
pub(crate) fn provision(slots: u32) -> Result<()> {
    checked(|_| {
        ensure!(slots == 4, "Q5 provision case mismatch");
        Ok(())
    })
}
pub(crate) fn lengths(prompt: usize, output: usize) -> Result<()> {
    checked(|r| check_lengths(r.limits, prompt, output))
}
fn check_lengths(l: Limits, prompt: usize, output: usize) -> Result<()> {
    ensure!(
        prompt > 0
            && prompt <= l.max_prompt_tokens
            && output > 0
            && output <= l.max_output_tokens
            && prompt.checked_add(output).is_some_and(|n| n <= 262144),
        "Q5 request exceeds reviewed prompt/output/context"
    );
    Ok(())
}
pub(crate) fn enqueue(
    prompt: &Vec<u32>,
    p: &SamplingParams,
    vision: bool,
    embed: bool,
) -> Result<()> {
    checked(|r| {
        check_lengths(r.limits, prompt.len(), p.max_tokens)?;
        ensure!(
            prompt.capacity() <= 262144
                && !vision
                && !embed
                && !p.logprobs
                && p.top_logprobs == 0
                && p.vision_fingerprint.is_none(),
            "Q5 unsupported request capacity/vision/logprobs"
        );
        // Existing sampling/tool grammar is unchanged. This bounds the admitted
        // request population; CPU policy remains an explicit operating allowance.
        r.attempts
            .fetch_update(Ordering::AcqRel, Ordering::Acquire, |n| {
                n.checked_add(1)
                    .filter(|&x| x <= r.limits.max_enqueue_attempts)
            })
            .map_err(|_| anyhow::anyhow!("Q5 finite enqueue-attempt ceiling"))?;
        Ok(())
    })?;
    health()
}
pub(crate) fn refuse_embedding() -> Result<()> {
    checked(|_| anyhow::bail!("Q5 selected case is text generation only"))
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn actual_request_bounds_do_not_change_workload() {
        let l = Limits {
            queue_capacity: 4,
            max_enqueue_attempts: 28,
            max_prompt_tokens: 262144,
            max_output_tokens: 8192,
        };
        check_lengths(l, 171954, 8192).unwrap();
        for (a, b) in [(0, 1), (262144, 1), (1, 8193), (usize::MAX, 1)] {
            assert!(check_lengths(l, a, b).is_err());
        }
    }
    #[test]
    fn actual_serve_binding_requires_real_queue_and_four_slot_mode() {
        let l = Limits {
            queue_capacity: 32,
            max_enqueue_attempts: 28,
            max_prompt_tokens: 262144,
            max_output_tokens: 8192,
        };
        check_serve(l, 32, Some(4)).unwrap();
        for (q, n) in [(31, Some(4)), (32, None), (32, Some(3))] {
            assert!(check_serve(l, q, n).is_err());
        }
        assert!(require_actual(l).is_err());
    }
}
pub(crate) fn record_failure() {
    if let Some(r) = ACTIVE.get() {
        r.failed.store(true, Ordering::Release);
    }
}
pub(crate) fn device(device: &mlx_native::MlxDevice) -> Result<()> {
    checked(|r| {
        ensure!(
            device.registry_id() == r.policy.device_registry_id
                && device.metal_device().max_buffer_length() as u64 == r.policy.max_buffer_bytes,
            "Q5 actual model device differs from reviewed capability"
        );
        Ok(())
    })
}
/// Acquires BEFORE handlers parse bodies/compile grammars; held through SSE body
/// disposal, not merely response headers. The existing 8MiB body limit remains.
pub(crate) async fn ingress(
    req: axum::extract::Request,
    next: axum::middleware::Next,
) -> axum::response::Response {
    use axum::{http::StatusCode, response::IntoResponse};
    let Some(r) = ACTIVE.get() else {
        return next.run(req).await;
    };
    if r.failed.load(Ordering::Acquire) {
        return (StatusCode::SERVICE_UNAVAILABLE, "Q5 selected case failed").into_response();
    }
    if !allowed_endpoint(req.method(), req.uri().path()) {
        return (
            StatusCode::BAD_REQUEST,
            "Q5 selected case endpoint unsupported",
        )
            .into_response();
    }
    let permit = match r.ingress.clone().try_acquire_owned() {
        Ok(p) => p,
        Err(_) => {
            return (
                StatusCode::TOO_MANY_REQUESTS,
                "Q5 bounded preparse ingress full",
            )
                .into_response()
        }
    };
    let response = next.run(req).await;
    hold_ingress_response(response, permit)
}
fn hold_ingress_response(
    response: axum::response::Response,
    permit: tokio::sync::OwnedSemaphorePermit,
) -> axum::response::Response {
    use futures_util::StreamExt;
    let (parts, body) = response.into_parts();
    let stream = body.into_data_stream().map(move |chunk| {
        let _ = &permit;
        chunk
    });
    axum::response::Response::from_parts(parts, axum::body::Body::from_stream(stream))
}

pub(crate) fn decoded_bytes(current: usize, next: usize) -> Result<()> {
    checked(|r| {
        ensure!(
            current
                .checked_add(next)
                .is_some_and(|n| n as u64 <= r.policy.max_decoded_bytes),
            "Q5 cumulative decoded output ceiling"
        );
        Ok(())
    })
}
pub(crate) fn is_active() -> bool {
    ACTIVE.get().is_some()
}
#[derive(Serialize)]
pub(crate) struct Facts {
    scope: &'static str,
    limits: Limits,
    policy: Policy,
    source_mandatory_minimum_not_peak:
        crate::inference::models::deepseek4::selected_budget_plan::Minimum,
    initial_model_absent_os_working_sample: operating_os::Sample,
    enqueues: u64,
    elapsed_ms: u128,
    failed: bool,
    physical_admission: bool,
}
pub(crate) fn facts() -> Option<Facts> {
    ACTIVE.get().map(|r| Facts {
        scope: "OneGuardedOperatingAttempt_MayRefuseAtCap_NotCompletionOrPhysicalProof",
        limits: r.limits,
        policy: r.policy,
        source_mandatory_minimum_not_peak: r.minimum,
        initial_model_absent_os_working_sample: r.initial_os,
        enqueues: r.attempts.load(Ordering::Acquire),
        elapsed_ms: r.started.elapsed().as_millis(),
        failed: r.failed.load(Ordering::Acquire),
        physical_admission: false,
    })
}

/// All actual selected application submission sites call this while roots are
/// still owned. Healthy default processes pay no extra waits or kernel work.
pub(crate) fn before_submit() -> Result<()> {
    if ACTIVE.get().is_some() {
        ensure!(
            super::collector::active(),
            "Q5 selected submission has no owning collector"
        );
        super::collector::checked_health()?;
    }
    Ok(())
}

fn allowed_endpoint(method: &axum::http::Method, path: &str) -> bool {
    (method == axum::http::Method::GET
        && matches!(
            path,
            "/health" | "/readyz" | "/metrics" | "/hf2q/v1/runtime" | "/v1/models"
        ))
        || (method == axum::http::Method::POST
            && matches!(path, "/v1/chat/completions" | "/shutdown"))
}
#[cfg(test)]
mod ingress_tests {
    #[test]
    fn allowed_get_and_post_responses_retain_the_same_capacity_until_drop() {
        for (method, path) in [
            (axum::http::Method::GET, "/metrics"),
            (axum::http::Method::POST, "/v1/chat/completions"),
        ] {
            assert!(super::allowed_endpoint(&method, path));
            let semaphore = std::sync::Arc::new(tokio::sync::Semaphore::new(1));
            let permit = semaphore.clone().try_acquire_owned().unwrap();
            let response = super::hold_ingress_response(
                axum::response::Response::new(axum::body::Body::from("retained")),
                permit,
            );
            assert!(semaphore.clone().try_acquire_owned().is_err());
            drop(response);
            assert!(semaphore.try_acquire_owned().is_ok());
        }
        assert!(!super::allowed_endpoint(
            &axum::http::Method::GET,
            "/v1/chat/completions"
        ));
        assert!(!super::allowed_endpoint(
            &axum::http::Method::POST,
            "/v1/models"
        ));
    }
}
