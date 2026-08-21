//! Source-bound, pre-model-weight/Metal-allocation work preflight for the
//! completed Qwen teacher.
//!
//! This capability consumes the exact B2a topology and teacher-evaluation
//! plan before any B2b Metal allocation. It proves source identity, vocabulary,
//! target dimensions, and checked execution-work arithmetic only. It neither
//! prepares weights nor executes the graph.

use anyhow::{ensure, Context, Result};
use serde::Serialize;
use sha2::{Digest, Sha256};
use std::path::Path;

use crate::intelligence::calibration::{
    RenderMode, TeacherPredictionPointKind, VerifiedTeacherPredictionPlan,
};
use crate::intelligence::exact_teacher::{
    preflight_structural_teacher_target, TeacherTargetArtifactLimits,
    EXACT_TEACHER_GREEDY_TOKEN_COUNT,
};

use super::topology::VerifiedQwen35Bf16TopologyV1;

const EXECUTION_PLAN_SCHEMA_VERSION: u32 = 1;
const EXECUTION_PLAN_PROFILE: &str = "dense_qwen35_source_teacher_work_preflight_v1";
const MIN_FRESH_PREFILL_TOKENS: usize = 16;
const HARD_MAX_EXAMPLES: usize = 4_096;
const HARD_MAX_FORWARD_CALLS: u64 = 1_000_000;
const HARD_MAX_INPUT_TOKENS_PROCESSED: u64 = 16 * 1024 * 1024;
const HARD_MAX_OUTPUT_HEAD_EVALUATIONS: u64 = 1_000_000;
const HARD_MAX_CACHE_TOKENS: usize = 4_096;

/// Caller limits for source-teacher execution work. These are checked against
/// both hard v1 ceilings and the exact opaque prediction plan.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub(crate) struct Qwen35SourceTeacherRunLimitsV1 {
    pub max_examples: usize,
    pub max_forward_calls: u64,
    pub max_input_tokens_processed: u64,
    pub max_output_head_evaluations: u64,
    pub max_cache_tokens: usize,
}

/// Exact work that the completed runner must observe before it can promote a
/// structural target to family-owned teacher authority.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub(super) struct Qwen35SourceTeacherExpectedWorkV1 {
    pub(super) example_count: usize,
    pub(super) completed_transcript_count: usize,
    pub(super) generation_prompt_count: usize,
    pub(super) prediction_row_count: usize,
    pub(super) forward_call_count: u64,
    pub(super) input_tokens_processed: u64,
    pub(super) output_head_evaluation_count: u64,
    pub(super) max_cache_tokens: usize,
    pub(super) target_artifact_bytes: u64,
    pub(super) greedy_token_count_per_prompt: usize,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
struct Qwen35SourceTeacherWorkReceiptV1 {
    schema_version: u32,
    profile: &'static str,
    source: crate::intelligence::measured_auto_quant::SourceIdentity,
    verified_source_manifest_sha256: String,
    topology_sha256: String,
    prediction_plan_sha256: String,
    target_limits: TeacherTargetArtifactLimits,
    run_limits: Qwen35SourceTeacherRunLimitsV1,
    vocabulary_size: usize,
    expected_work: Qwen35SourceTeacherExpectedWorkV1,
    work_plan_sha256: String,
}

#[derive(Serialize)]
struct WorkHashView<'a> {
    schema_version: u32,
    profile: &'static str,
    source: &'a crate::intelligence::measured_auto_quant::SourceIdentity,
    verified_source_manifest_sha256: &'a str,
    topology_sha256: &'a str,
    prediction_plan_sha256: &'a str,
    target_limits: TeacherTargetArtifactLimits,
    run_limits: Qwen35SourceTeacherRunLimitsV1,
    vocabulary_size: usize,
    expected_work: Qwen35SourceTeacherExpectedWorkV1,
}

/// Opaque proof that source, plan, vocabulary, target framing, and execution
/// work were joined before any model or Metal allocation.
pub(crate) struct StructurallyBoundQwen35SourceTeacherWorkV1 {
    topology: VerifiedQwen35Bf16TopologyV1,
    prediction_plan: VerifiedTeacherPredictionPlan,
    receipt: Qwen35SourceTeacherWorkReceiptV1,
}

impl StructurallyBoundQwen35SourceTeacherWorkV1 {
    pub(crate) fn work_plan_sha256(&self) -> &str {
        &self.receipt.work_plan_sha256
    }

    pub(crate) fn forward_call_count(&self) -> u64 {
        self.receipt.expected_work.forward_call_count
    }

    pub(crate) fn max_cache_tokens(&self) -> usize {
        self.receipt.expected_work.max_cache_tokens
    }

    pub(crate) fn example_count(&self) -> usize {
        self.receipt.expected_work.example_count
    }

    pub(crate) fn prediction_row_count(&self) -> usize {
        self.receipt.expected_work.prediction_row_count
    }

    pub(crate) fn input_tokens_processed(&self) -> u64 {
        self.receipt.expected_work.input_tokens_processed
    }

    pub(crate) fn output_head_evaluation_count(&self) -> u64 {
        self.receipt.expected_work.output_head_evaluation_count
    }

    pub(crate) fn target_artifact_bytes(&self) -> u64 {
        self.receipt.expected_work.target_artifact_bytes
    }

    pub(super) fn preparation_parts(
        &self,
    ) -> (
        &VerifiedQwen35Bf16TopologyV1,
        Qwen35SourceTeacherExpectedWorkV1,
    ) {
        (&self.topology, self.receipt.expected_work)
    }

    /// Exercise the exact no-clobber target reservation and immediately drop
    /// the private temp. This is a destination dry-run only; the consuming
    /// production transition reserves again before weight allocation.
    pub(crate) fn preflight_target_destination(&self, output: &Path) -> Result<()> {
        let reservation = preflight_structural_teacher_target(
            &self.prediction_plan,
            self.receipt.vocabulary_size,
            self.receipt.target_limits,
        )?
        .reserve(output)?;
        reservation.validate_private()?;
        Ok(())
    }

    pub(super) fn into_parts(
        self,
    ) -> (
        VerifiedQwen35Bf16TopologyV1,
        VerifiedTeacherPredictionPlan,
        TeacherTargetArtifactLimits,
        Qwen35SourceTeacherRunLimitsV1,
        Qwen35SourceTeacherExpectedWorkV1,
        String,
    ) {
        (
            self.topology,
            self.prediction_plan,
            self.receipt.target_limits,
            self.receipt.run_limits,
            self.receipt.expected_work,
            self.receipt.work_plan_sha256,
        )
    }
}

pub(crate) fn preflight_qwen35_source_teacher_execution(
    topology: VerifiedQwen35Bf16TopologyV1,
    prediction_plan: VerifiedTeacherPredictionPlan,
    target_limits: TeacherTargetArtifactLimits,
    run_limits: Qwen35SourceTeacherRunLimitsV1,
) -> Result<StructurallyBoundQwen35SourceTeacherWorkV1> {
    validate_run_limits(run_limits)?;
    let manifest = prediction_plan.manifest();
    ensure!(
        &manifest.source == topology.source()
            && manifest.verified_source_manifest_sha256
                == topology.verified_source_manifest_sha256(),
        "source teacher prediction plan does not match the retained source topology"
    );
    let config = topology.projected_config_for_teacher()?;
    let vocabulary_size = usize::try_from(config.vocab_size)
        .context("source teacher vocabulary is not representable")?;
    let target_preflight =
        preflight_structural_teacher_target(&prediction_plan, vocabulary_size, target_limits)
            .context("source teacher target preflight failed")?;

    let mut completed_transcript_count = 0usize;
    let mut generation_prompt_count = 0usize;
    let mut forward_call_count = 0u64;
    let mut input_tokens_processed = 0u64;
    let mut output_head_evaluation_count = 0u64;
    let mut max_cache_tokens = 0usize;
    prediction_plan.visit_examples(|example, _tokens, points, greedy| -> Result<()> {
        let first = points
            .first()
            .context("source teacher example has no point")?;
        ensure!(
            first.prefix_token_count >= MIN_FRESH_PREFILL_TOKENS,
            "source teacher v1 requires a fresh prefix of at least 16 tokens"
        );
        match example.render_mode {
            RenderMode::CompletedAssistantTranscript => {
                ensure!(greedy.is_none(), "completed transcript has a greedy prompt");
                ensure!(
                    points.iter().all(|point| matches!(
                        point.kind,
                        TeacherPredictionPointKind::TeacherForced { .. }
                    )),
                    "completed transcript contains a non-teacher-forced point"
                );
                let last = points.last().unwrap().prefix_token_count;
                let calls = last
                    .checked_sub(first.prefix_token_count)
                    .and_then(|delta| delta.checked_add(1))
                    .context("source teacher transcript call count overflow")?;
                checked_add(
                    &mut forward_call_count,
                    u64::try_from(calls).context("transcript call count is not representable")?,
                    "forward calls",
                )?;
                checked_add(
                    &mut input_tokens_processed,
                    u64::try_from(last)
                        .context("transcript input-token count is not representable")?,
                    "input tokens",
                )?;
                checked_add(
                    &mut output_head_evaluation_count,
                    u64::try_from(points.len())
                        .context("transcript output-head count is not representable")?,
                    "output-head evaluations",
                )?;
                max_cache_tokens = max_cache_tokens.max(last);
                completed_transcript_count += 1;
            }
            RenderMode::GenerationPrompt => {
                ensure!(
                    points.len() == 1
                        && matches!(points[0].kind, TeacherPredictionPointKind::GenerationNext)
                        && greedy.is_some(),
                    "generation prompt does not have its exact next-token and greedy contract"
                );
                let cache_tokens = first
                    .prefix_token_count
                    .checked_add(EXACT_TEACHER_GREEDY_TOKEN_COUNT - 1)
                    .context("source teacher greedy cache count overflow")?;
                checked_add(
                    &mut forward_call_count,
                    u64::try_from(EXACT_TEACHER_GREEDY_TOKEN_COUNT)
                        .context("greedy call count is not representable")?,
                    "forward calls",
                )?;
                checked_add(
                    &mut input_tokens_processed,
                    u64::try_from(cache_tokens)
                        .context("greedy input-token count is not representable")?,
                    "input tokens",
                )?;
                checked_add(
                    &mut output_head_evaluation_count,
                    u64::try_from(EXACT_TEACHER_GREEDY_TOKEN_COUNT)
                        .context("greedy output-head count is not representable")?,
                    "output-head evaluations",
                )?;
                max_cache_tokens = max_cache_tokens.max(cache_tokens);
                generation_prompt_count += 1;
            }
        }
        Ok(())
    })?;

    ensure!(
        manifest.total_example_count <= run_limits.max_examples
            && forward_call_count <= run_limits.max_forward_calls
            && input_tokens_processed <= run_limits.max_input_tokens_processed
            && output_head_evaluation_count <= run_limits.max_output_head_evaluations
            && max_cache_tokens <= run_limits.max_cache_tokens,
        "source teacher exact work exceeds its requested limits"
    );
    ensure!(
        max_cache_tokens
            <= usize::try_from(config.max_position_embeddings)
                .context("source teacher context limit is not representable")?,
        "source teacher exact cache work exceeds the authenticated model context"
    );
    let expected_work = Qwen35SourceTeacherExpectedWorkV1 {
        example_count: manifest.total_example_count,
        completed_transcript_count,
        generation_prompt_count,
        prediction_row_count: prediction_plan.prediction_point_count(),
        forward_call_count,
        input_tokens_processed,
        output_head_evaluation_count,
        max_cache_tokens,
        target_artifact_bytes: target_preflight.preflight_bytes(),
        greedy_token_count_per_prompt: EXACT_TEACHER_GREEDY_TOKEN_COUNT,
    };
    let mut receipt = Qwen35SourceTeacherWorkReceiptV1 {
        schema_version: EXECUTION_PLAN_SCHEMA_VERSION,
        profile: EXECUTION_PLAN_PROFILE,
        source: manifest.source.clone(),
        verified_source_manifest_sha256: manifest.verified_source_manifest_sha256.clone(),
        topology_sha256: topology.topology_sha256().into(),
        prediction_plan_sha256: manifest.manifest_sha256.clone(),
        target_limits,
        run_limits,
        vocabulary_size,
        expected_work,
        work_plan_sha256: String::new(),
    };
    receipt.work_plan_sha256 = hash_receipt(&receipt)?;
    Ok(StructurallyBoundQwen35SourceTeacherWorkV1 {
        topology,
        prediction_plan,
        receipt,
    })
}

fn validate_run_limits(limits: Qwen35SourceTeacherRunLimitsV1) -> Result<()> {
    ensure!(
        limits.max_examples > 0
            && limits.max_examples <= HARD_MAX_EXAMPLES
            && limits.max_forward_calls > 0
            && limits.max_forward_calls <= HARD_MAX_FORWARD_CALLS
            && limits.max_input_tokens_processed > 0
            && limits.max_input_tokens_processed <= HARD_MAX_INPUT_TOKENS_PROCESSED
            && limits.max_output_head_evaluations > 0
            && limits.max_output_head_evaluations <= HARD_MAX_OUTPUT_HEAD_EVALUATIONS
            && limits.max_cache_tokens > 0
            && limits.max_cache_tokens <= HARD_MAX_CACHE_TOKENS,
        "source teacher run limits exceed the hard v1 envelope"
    );
    Ok(())
}

fn checked_add(total: &mut u64, value: u64, label: &str) -> Result<()> {
    *total = total
        .checked_add(value)
        .with_context(|| format!("source teacher {label} overflow"))?;
    Ok(())
}

fn hash_receipt(receipt: &Qwen35SourceTeacherWorkReceiptV1) -> Result<String> {
    let view = WorkHashView {
        schema_version: receipt.schema_version,
        profile: receipt.profile,
        source: &receipt.source,
        verified_source_manifest_sha256: &receipt.verified_source_manifest_sha256,
        topology_sha256: &receipt.topology_sha256,
        prediction_plan_sha256: &receipt.prediction_plan_sha256,
        target_limits: receipt.target_limits,
        run_limits: receipt.run_limits,
        vocabulary_size: receipt.vocabulary_size,
        expected_work: receipt.expected_work,
    };
    Ok(hex::encode(Sha256::digest(serde_json::to_vec(&view)?)))
}
