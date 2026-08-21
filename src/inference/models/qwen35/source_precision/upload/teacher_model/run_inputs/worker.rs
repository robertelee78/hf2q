//! One-shot consuming source-teacher transaction.

use std::any::Any;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::path::Path;

use anyhow::{anyhow, ensure, Context, Result};

use crate::inference::models::qwen35::execution_dispatch::{
    source_teacher_graph_policy_sha256, with_source_teacher_graph_scope, SourceTeacherGraphScope,
};
use crate::inference::models::qwen35::source_precision::snapshot::VerifiedQwenSourceSnapshot;
use crate::intelligence::calibration::{
    RenderMode, TeacherPredictionPointReceipt, VerifiedCalibrationPredictionPlan,
};
use crate::intelligence::exact_teacher::{
    canonical_teacher_argmax, StructurallyVerifiedTeacherTargetArtifact,
    UnpublishedStructuralTeacherTargetArtifact, EXACT_TEACHER_GREEDY_TOKEN_COUNT,
};

use super::super::runner::{SourceTeacherCallResult, SourceTeacherSessionV1};
use super::PreparedQwen35SourceTeacherRunInputsV1;

mod parts;
mod receipt;
mod schedule;

use parts::SourceTeacherWorkerPartsV1;
use receipt::{
    build_completion_receipt, Qwen35SourceTeacherCompletionReceiptV1,
    Qwen35SourceTeacherObservedWorkV1,
};
use schedule::{expected_call_schedule_sha256, CallScheduleHasher};

const WORKER_NAME: &str = "hf2q-qwen35-source-teacher-v1";

/// Opaque completed source-teacher authority.
///
/// The retained source snapshot and published target inode stay live. No raw
/// source file, Metal buffer, cache, model, session, or logit row escapes.
pub(crate) struct VerifiedQwen35SourceTeacherTargetV1 {
    _source_snapshot: VerifiedQwenSourceSnapshot,
    _prediction_plan: VerifiedCalibrationPredictionPlan,
    target: StructurallyVerifiedTeacherTargetArtifact,
    receipt: Qwen35SourceTeacherCompletionReceiptV1,
}

impl VerifiedQwen35SourceTeacherTargetV1 {
    pub(crate) fn path(&self) -> &Path {
        self.target.path()
    }

    pub(crate) fn target_artifact_sha256(&self) -> &str {
        &self.target.receipt().target_artifact.sha256
    }

    pub(crate) fn completion_receipt_sha256(&self) -> &str {
        self.receipt.receipt_sha256()
    }
}

struct ReadyToPublishQwen35SourceTeacherTargetV1 {
    source_snapshot: VerifiedQwenSourceSnapshot,
    prediction_plan: VerifiedCalibrationPredictionPlan,
    target: UnpublishedStructuralTeacherTargetArtifact,
    receipt: Qwen35SourceTeacherCompletionReceiptV1,
}

struct CompletedExecutionV1 {
    observed: Qwen35SourceTeacherObservedWorkV1,
    graph_policy_sha256: String,
    call_schedule_sha256: String,
}

#[derive(Clone, Default)]
struct WorkerBehavior {
    panic_after_completed_call: Option<u64>,
    #[cfg(test)]
    error_drain_directory: Option<std::path::PathBuf>,
}

impl WorkerBehavior {
    fn after_completed_call(&self, completed_call_count: u64) {
        if self.panic_after_completed_call == Some(completed_call_count) {
            panic!("injected source-teacher worker panic after completed call");
        }
    }

    fn after_error_drain(&self) -> Result<()> {
        #[cfg(test)]
        if let Some(directory) = &self.error_drain_directory {
            let entries = std::fs::read_dir(directory)?.collect::<std::io::Result<Vec<_>>>()?;
            ensure!(
                entries.len() == 1,
                "private target was not retained through the terminal drain"
            );
        }
        Ok(())
    }
}

/// Consume sealed run inputs on a fresh named thread and publish only after a
/// successful join. The no-replace rename is the final fallible operation.
pub(crate) fn run_qwen35_source_teacher(
    inputs: PreparedQwen35SourceTeacherRunInputsV1,
) -> Result<VerifiedQwen35SourceTeacherTargetV1> {
    run_qwen35_source_teacher_with(inputs, WorkerBehavior::default(), || Ok(()))
}

fn run_qwen35_source_teacher_with<BeforePublish>(
    inputs: PreparedQwen35SourceTeacherRunInputsV1,
    behavior: WorkerBehavior,
    before_publish: BeforePublish,
) -> Result<VerifiedQwen35SourceTeacherTargetV1>
where
    BeforePublish: FnOnce() -> Result<()>,
{
    let worker = std::thread::Builder::new()
        .name(WORKER_NAME.into())
        .spawn(move || worker_main(inputs, behavior))
        .context("spawn source-teacher worker")?;
    let ready = worker.join().map_err(|panic| {
        anyhow!(
            "source-teacher worker panicked outside its guard: {}",
            panic_text(panic)
        )
    })??;
    before_publish()?;
    let target = ready.target.publish_noclobber()?;
    Ok(VerifiedQwen35SourceTeacherTargetV1 {
        _source_snapshot: ready.source_snapshot,
        _prediction_plan: ready.prediction_plan,
        target,
        receipt: ready.receipt,
    })
}

#[cfg(test)]
fn run_qwen35_source_teacher_with_behavior_for_test(
    inputs: PreparedQwen35SourceTeacherRunInputsV1,
    behavior: WorkerBehavior,
    before_publish: impl FnOnce() -> Result<()>,
) -> Result<VerifiedQwen35SourceTeacherTargetV1> {
    run_qwen35_source_teacher_with(inputs, behavior, before_publish)
}

fn worker_main(
    inputs: PreparedQwen35SourceTeacherRunInputsV1,
    behavior: WorkerBehavior,
) -> Result<ReadyToPublishQwen35SourceTeacherTargetV1> {
    ensure!(
        std::thread::current().name() == Some(WORKER_NAME),
        "source-teacher execution is not on its dedicated worker"
    );
    let parts = inputs.into_worker_parts()?;
    with_source_teacher_graph_scope(|scope| worker_scoped(parts, scope, behavior))
}

fn worker_scoped(
    parts: SourceTeacherWorkerPartsV1,
    scope: &SourceTeacherGraphScope,
    behavior: WorkerBehavior,
) -> Result<ReadyToPublishQwen35SourceTeacherTargetV1> {
    let SourceTeacherWorkerPartsV1 {
        teacher,
        cache,
        prediction_plan,
        target_reservation,
        expected_work,
        receipt: predecessor,
    } = parts;
    let expected_source_snapshot_catalog_sha256 =
        teacher.receipt.source_snapshot_catalog_sha256.clone();
    let mut session = SourceTeacherSessionV1::new(scope, teacher, cache)?;
    ensure!(
        session.device_name() == predecessor.device_name
            && session.device_registry_id() == predecessor.device_registry_id,
        "source-teacher worker device differs from sealed run inputs"
    );
    let source_graph_policy_sha256 = source_teacher_graph_policy_sha256()?;
    let mut stream = target_reservation.begin(&prediction_plan)?;
    let outcome = catch_unwind(AssertUnwindSafe(|| {
        execute_plan(&mut session, &prediction_plan, &mut stream, &behavior)
    }));
    let completed = match outcome {
        Ok(Ok(completed)) => completed,
        Ok(Err(error)) => {
            let drain = session.terminal_drain_after_panic();
            return match drain {
                Ok(()) => {
                    behavior.after_error_drain()?;
                    Err(error)
                }
                Err(drain_error) => Err(anyhow!(
                    "source-teacher execution failed: {error:#}; terminal drain failed: {drain_error:#}"
                )),
            };
        }
        Err(panic) => {
            let message = panic_text(panic);
            let drain = session.terminal_drain_after_panic();
            return match drain {
                Ok(()) => {
                    behavior.after_error_drain()?;
                    Err(anyhow!("source-teacher execution panicked: {message}"))
                }
                Err(drain_error) => Err(anyhow!(
                    "source-teacher execution panicked: {message}; terminal drain failed: {drain_error:#}"
                )),
            };
        }
    };
    session.terminal_drain_for_completion()?;
    ensure!(
        completed.observed == observed_from_expected(expected_work),
        "completed source-teacher counters differ from sealed expected work"
    );
    let target = stream.finish_unpublished()?;
    let expected_schedule = expected_call_schedule_sha256(&prediction_plan, target.receipt())?;
    ensure!(
        expected_schedule == completed.call_schedule_sha256
            && completed.graph_policy_sha256 == source_graph_policy_sha256,
        "completed source-teacher call schedule or graph policy differs"
    );
    let source_snapshot = session.finish_source_lineage()?;
    ensure!(
        source_snapshot.catalog_sha256() == expected_source_snapshot_catalog_sha256,
        "completed source snapshot differs from the prepared teacher"
    );
    let completion_receipt = build_completion_receipt(
        &predecessor,
        source_snapshot.catalog_sha256().into(),
        source_graph_policy_sha256,
        completed.call_schedule_sha256,
        completed.observed,
        target.receipt(),
    )?;
    Ok(ReadyToPublishQwen35SourceTeacherTargetV1 {
        source_snapshot,
        prediction_plan,
        target,
        receipt: completion_receipt,
    })
}

fn execute_plan(
    session: &mut SourceTeacherSessionV1<'_>,
    plan: &VerifiedCalibrationPredictionPlan,
    stream: &mut crate::intelligence::exact_teacher::StructuralTeacherTargetStream<'_>,
    behavior: &WorkerBehavior,
) -> Result<CompletedExecutionV1> {
    let mut observed = Qwen35SourceTeacherObservedWorkV1::default();
    let mut graph_policy_sha256 = None;
    let mut schedule = CallScheduleHasher::new();
    plan.visit_examples(|example, tokens, points, greedy| -> Result<()> {
        session.reset_example()?;
        observed.cache_reset_count += 1;
        observed.example_count += 1;
        match example.render_mode {
            RenderMode::CompletedAssistantTranscript => {
                ensure!(greedy.is_none(), "completed transcript has a greedy prompt");
                execute_completed_transcript(
                    session,
                    stream,
                    &mut observed,
                    &mut graph_policy_sha256,
                    &mut schedule,
                    &example.stable_id,
                    tokens,
                    points,
                    behavior,
                )?;
                observed.completed_transcript_count += 1;
            }
            RenderMode::GenerationPrompt => {
                let prompt = greedy.context("generation example lacks greedy receipt")?;
                ensure!(
                    points.len() == 1,
                    "generation example does not have one row"
                );
                execute_generation(
                    session,
                    stream,
                    &mut observed,
                    &mut graph_policy_sha256,
                    &mut schedule,
                    &example.stable_id,
                    tokens,
                    &points[0],
                    prompt,
                    behavior,
                )?;
                observed.generation_prompt_count += 1;
            }
        }
        Ok(())
    })?;
    Ok(CompletedExecutionV1 {
        observed,
        graph_policy_sha256: graph_policy_sha256
            .context("source-teacher plan executed no calls")?,
        call_schedule_sha256: schedule.finish(),
    })
}

#[allow(clippy::too_many_arguments)]
fn execute_completed_transcript(
    session: &mut SourceTeacherSessionV1<'_>,
    stream: &mut crate::intelligence::exact_teacher::StructuralTeacherTargetStream<'_>,
    observed: &mut Qwen35SourceTeacherObservedWorkV1,
    graph_policy: &mut Option<String>,
    schedule: &mut CallScheduleHasher,
    stable_id: &str,
    tokens: &[u32],
    points: &[TeacherPredictionPointReceipt],
    behavior: &WorkerBehavior,
) -> Result<()> {
    let first = points.first().context("completed transcript has no row")?;
    let last_prefix = points.last().unwrap().prefix_token_count;
    ensure!(
        first.prefix_token_count <= last_prefix && last_prefix <= tokens.len(),
        "completed transcript prefixes exceed retained tokens"
    );
    let mut next_point = 0usize;
    for prefix in first.prefix_token_count..=last_prefix {
        let (first_position, input) = if prefix == first.prefix_token_count {
            (0_u32, &tokens[..prefix])
        } else {
            (u32::try_from(prefix - 1)?, &tokens[prefix - 1..prefix])
        };
        let emit = points
            .get(next_point)
            .is_some_and(|point| point.prefix_token_count == prefix);
        let result = session.run_call(input, emit)?;
        record_completed_call(
            session,
            observed,
            graph_policy,
            schedule,
            stable_id,
            first_position,
            input,
            emit,
            result.as_ref(),
            behavior,
        )?;
        if emit {
            let result = result.context("source-teacher row call returned no logits")?;
            stream.write_row(&points[next_point], &result.logits)?;
            observed.prediction_row_count += 1;
            next_point += 1;
        }
    }
    ensure!(
        next_point == points.len(),
        "completed transcript skipped a row"
    );
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn execute_generation(
    session: &mut SourceTeacherSessionV1<'_>,
    stream: &mut crate::intelligence::exact_teacher::StructuralTeacherTargetStream<'_>,
    observed: &mut Qwen35SourceTeacherObservedWorkV1,
    graph_policy: &mut Option<String>,
    schedule: &mut CallScheduleHasher,
    stable_id: &str,
    tokens: &[u32],
    point: &TeacherPredictionPointReceipt,
    prompt: &crate::intelligence::calibration::TeacherGreedyPromptReceipt,
    behavior: &WorkerBehavior,
) -> Result<()> {
    ensure!(
        point.prefix_token_count == tokens.len(),
        "generation prefix differs from retained prompt"
    );
    let result = session
        .run_call(tokens, true)?
        .context("generation prompt returned no logits")?;
    record_completed_call(
        session,
        observed,
        graph_policy,
        schedule,
        stable_id,
        0,
        tokens,
        true,
        Some(&result),
        behavior,
    )?;
    let first_token = stream.write_row(point, &result.logits)?;
    observed.prediction_row_count += 1;
    let mut trajectory = Vec::with_capacity(EXACT_TEACHER_GREEDY_TOKEN_COUNT);
    trajectory.push(first_token);
    for _ in 1..EXACT_TEACHER_GREEDY_TOKEN_COUNT {
        let token = *trajectory.last().unwrap();
        let first_position = u32::try_from(session.next_position())?;
        let result = session
            .run_call(std::slice::from_ref(&token), true)?
            .context("generation continuation returned no logits")?;
        record_completed_call(
            session,
            observed,
            graph_policy,
            schedule,
            stable_id,
            first_position,
            std::slice::from_ref(&token),
            true,
            Some(&result),
            behavior,
        )?;
        trajectory.push(canonical_teacher_argmax(&result.logits)?);
    }
    stream.write_trajectory(prompt, &trajectory)?;
    observed.trajectory_count += 1;
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn record_completed_call(
    session: &SourceTeacherSessionV1<'_>,
    observed: &mut Qwen35SourceTeacherObservedWorkV1,
    graph_policy: &mut Option<String>,
    schedule: &mut CallScheduleHasher,
    stable_id: &str,
    first_position: u32,
    input: &[u32],
    emit: bool,
    result: Option<&SourceTeacherCallResult>,
    behavior: &WorkerBehavior,
) -> Result<()> {
    ensure!(
        emit == result.is_some(),
        "source-teacher output-head mode differs"
    );
    if let Some(result) = result {
        match graph_policy {
            Some(expected) => ensure!(
                expected == &result.graph_policy_sha256,
                "source-teacher graph policy changed during execution"
            ),
            None => *graph_policy = Some(result.graph_policy_sha256.clone()),
        }
        observed.output_head_evaluation_count += 1;
    }
    observed.forward_call_count += 1;
    observed.input_tokens_processed = observed
        .input_tokens_processed
        .checked_add(u64::try_from(input.len())?)
        .context("source-teacher observed input-token overflow")?;
    observed.terminal_call_completion_count += 1;
    observed.max_cache_tokens = observed
        .max_cache_tokens
        .max(usize::try_from(session.next_position())?);
    schedule.record(stable_id, first_position, input, emit)?;
    behavior.after_completed_call(observed.forward_call_count);
    Ok(())
}

fn observed_from_expected(
    expected: crate::inference::models::qwen35::source_precision::teacher_execution_plan::Qwen35SourceTeacherExpectedWorkV1,
) -> Qwen35SourceTeacherObservedWorkV1 {
    Qwen35SourceTeacherObservedWorkV1 {
        example_count: expected.example_count,
        completed_transcript_count: expected.completed_transcript_count,
        generation_prompt_count: expected.generation_prompt_count,
        prediction_row_count: expected.prediction_row_count,
        trajectory_count: expected.generation_prompt_count,
        cache_reset_count: expected.example_count,
        forward_call_count: expected.forward_call_count,
        input_tokens_processed: expected.input_tokens_processed,
        output_head_evaluation_count: expected.output_head_evaluation_count,
        terminal_call_completion_count: expected.forward_call_count,
        max_cache_tokens: expected.max_cache_tokens,
    }
}

fn panic_text(panic: Box<dyn Any + Send>) -> String {
    panic
        .downcast_ref::<&str>()
        .map(|value| (*value).to_owned())
        .or_else(|| panic.downcast_ref::<String>().cloned())
        .unwrap_or_else(|| "non-string panic payload".into())
}

#[cfg(test)]
mod tests;
