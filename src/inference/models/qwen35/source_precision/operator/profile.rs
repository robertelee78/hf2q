use anyhow::{ensure, Context, Result};
use serde::Deserialize;
use sha2::{Digest, Sha256};

use crate::intelligence::calibration::TeacherPredictionPlanLimits;
use crate::intelligence::exact_teacher::TeacherTargetArtifactLimits;

use super::super::{
    Qwen35SourceTeacherPreparationPolicyV1, Qwen35SourceTeacherRunLimitsV1,
    QwenSourceMetalUploadLimits,
};

const PROFILE_BYTES: &[u8] = include_bytes!(
    "../../../../../../data/calibration/qwen38-source-teacher-canary-v1/profile.json"
);
pub(super) const PROFILE_SHA256: &str =
    "58ea26548ba581ec79191e44a59a2fd5b8274693baba69ab7067c7ea825c8c7e";

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct OfficialEvidenceProfileV1 {
    pub(super) schema_version: u32,
    pub(super) profile: String,
    pub(super) source: ProfileSourceV1,
    pub(super) dataset: ProfileDatasetV1,
    pub(super) render: ProfileRenderV1,
    prediction_limits: ProfilePredictionLimitsV1,
    target_limits: ProfileTargetLimitsV1,
    run_limits: ProfileRunLimitsV1,
    upload_limits: ProfileUploadLimitsV1,
    preparation_policy: ProfilePreparationPolicyV1,
    pub(super) scope: ProfileScopeV1,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct ProfileSourceV1 {
    pub(super) repository_id: String,
    pub(super) revision: String,
    pub(super) manifest_id: String,
    pub(super) manifest_sha256: String,
    pub(super) bundle_sha256: String,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct ProfileDatasetV1 {
    pub(super) dataset_id: String,
    pub(super) revision: String,
    pub(super) license: String,
    pub(super) seed: u64,
    pub(super) calibration_sha256: String,
    pub(super) policy_validation_sha256: String,
    pub(super) acceptance_holdout_sha256: String,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct ProfileRenderV1 {
    pub(super) renderer_revision: String,
    pub(super) max_tokens_per_example: usize,
    pub(super) token_window_size: usize,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ProfilePredictionLimitsV1 {
    max_examples: usize,
    max_total_tokens: usize,
    max_rendered_utf8_bytes: u64,
    max_prediction_points: usize,
    max_prefix_tokens: usize,
    max_generation_prompts: usize,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ProfileTargetLimitsV1 {
    max_vocabulary_size: usize,
    max_prediction_rows: usize,
    max_target_bytes: u64,
    top_k: usize,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ProfileRunLimitsV1 {
    max_examples: usize,
    max_forward_calls: u64,
    max_input_tokens_processed: u64,
    max_output_head_evaluations: u64,
    max_cache_tokens: usize,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ProfileUploadLimitsV1 {
    max_output_tensors: usize,
    max_total_output_bytes: u64,
    max_single_buffer_bytes: u64,
    host_reserve_bytes: u64,
    metal_reserve_bytes: u64,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ProfilePreparationPolicyV1 {
    max_cpu_control_mirror_bytes: u64,
    unmeasured_runtime_reserve_bytes: u64,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct ProfileScopeV1 {
    pub(super) canary_only: bool,
    pub(super) dynamic_calibration_sufficient: bool,
    pub(super) source_bf16_controls_f32: bool,
    pub(super) vision_executed: bool,
    pub(super) mtp_executed: bool,
    pub(super) tq: bool,
    pub(super) q4_repack: bool,
    pub(super) dwq: bool,
}

impl OfficialEvidenceProfileV1 {
    pub(super) fn prediction_limits(&self) -> TeacherPredictionPlanLimits {
        TeacherPredictionPlanLimits {
            max_examples: self.prediction_limits.max_examples,
            max_total_tokens: self.prediction_limits.max_total_tokens,
            max_rendered_utf8_bytes: self.prediction_limits.max_rendered_utf8_bytes,
            max_prediction_points: self.prediction_limits.max_prediction_points,
            max_prefix_tokens: self.prediction_limits.max_prefix_tokens,
            max_generation_prompts: self.prediction_limits.max_generation_prompts,
        }
    }

    pub(super) fn target_limits(&self) -> TeacherTargetArtifactLimits {
        TeacherTargetArtifactLimits {
            max_vocabulary_size: self.target_limits.max_vocabulary_size,
            max_prediction_rows: self.target_limits.max_prediction_rows,
            max_target_bytes: self.target_limits.max_target_bytes,
            top_k: self.target_limits.top_k,
        }
    }

    pub(super) fn run_limits(&self) -> Qwen35SourceTeacherRunLimitsV1 {
        Qwen35SourceTeacherRunLimitsV1 {
            max_examples: self.run_limits.max_examples,
            max_forward_calls: self.run_limits.max_forward_calls,
            max_input_tokens_processed: self.run_limits.max_input_tokens_processed,
            max_output_head_evaluations: self.run_limits.max_output_head_evaluations,
            max_cache_tokens: self.run_limits.max_cache_tokens,
        }
    }

    pub(super) fn upload_limits(&self) -> QwenSourceMetalUploadLimits {
        QwenSourceMetalUploadLimits {
            max_output_tensors: self.upload_limits.max_output_tensors,
            max_total_output_bytes: self.upload_limits.max_total_output_bytes,
            max_single_buffer_bytes: self.upload_limits.max_single_buffer_bytes,
            host_reserve_bytes: self.upload_limits.host_reserve_bytes,
            metal_reserve_bytes: self.upload_limits.metal_reserve_bytes,
        }
    }

    pub(super) fn preparation_policy(&self) -> Qwen35SourceTeacherPreparationPolicyV1 {
        Qwen35SourceTeacherPreparationPolicyV1 {
            max_cpu_control_mirror_bytes: self.preparation_policy.max_cpu_control_mirror_bytes,
            unmeasured_runtime_reserve_bytes: self
                .preparation_policy
                .unmeasured_runtime_reserve_bytes,
        }
    }
}

pub(super) fn official_profile() -> Result<OfficialEvidenceProfileV1> {
    ensure!(
        hex::encode(Sha256::digest(PROFILE_BYTES)) == PROFILE_SHA256,
        "embedded source-teacher profile bytes changed"
    );
    let profile: OfficialEvidenceProfileV1 =
        serde_json::from_slice(PROFILE_BYTES).context("parse embedded source-teacher profile")?;
    ensure!(
        profile.schema_version == 1
            && profile.profile == "qwen38_source_teacher_canary_v1"
            && profile.source.repository_id == "Qwen/Qwen3.8-27B"
            && profile.source.revision == "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
            && profile.scope.canary_only
            && !profile.scope.dynamic_calibration_sufficient
            && profile.scope.source_bf16_controls_f32
            && !profile.scope.vision_executed
            && !profile.scope.mtp_executed
            && !profile.scope.tq
            && !profile.scope.q4_repack
            && !profile.scope.dwq,
        "embedded source-teacher profile violates the canary/no-DWQ scope"
    );
    profile.upload_limits().validate()?;
    Ok(profile)
}

#[cfg(test)]
pub(super) fn profile_bytes_for_test() -> &'static [u8] {
    PROFILE_BYTES
}
