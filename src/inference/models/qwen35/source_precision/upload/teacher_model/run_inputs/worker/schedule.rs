//! Canonical completed-call schedule hashing.

use anyhow::{ensure, Context, Result};
use sha2::{Digest, Sha256};

use crate::intelligence::calibration::{RenderMode, VerifiedTeacherPredictionPlan};
use crate::intelligence::exact_teacher::ExactTeacherTargetReceipt;

const CALL_SCHEDULE_DOMAIN: &[u8] = b"hf2q-qwen35-source-teacher-call-schedule-v1";

pub(super) struct CallScheduleHasher(Sha256);

impl CallScheduleHasher {
    pub(super) fn new() -> Self {
        let mut hasher = Sha256::new();
        hasher.update(CALL_SCHEDULE_DOMAIN);
        Self(hasher)
    }

    pub(super) fn record(
        &mut self,
        stable_id: &str,
        first_position: u32,
        tokens: &[u32],
        emit: bool,
    ) -> Result<()> {
        self.0.update(u64::try_from(stable_id.len())?.to_le_bytes());
        self.0.update(stable_id.as_bytes());
        self.0.update(first_position.to_le_bytes());
        self.0.update(u64::try_from(tokens.len())?.to_le_bytes());
        for token in tokens {
            self.0.update(token.to_le_bytes());
        }
        self.0.update([u8::from(emit)]);
        Ok(())
    }

    pub(super) fn finish(self) -> String {
        hex::encode(self.0.finalize())
    }
}

pub(super) fn expected_call_schedule_sha256(
    plan: &VerifiedTeacherPredictionPlan,
    target: &ExactTeacherTargetReceipt,
) -> Result<String> {
    let mut schedule = CallScheduleHasher::new();
    let mut trajectory_index = 0usize;
    plan.visit_examples(|example, tokens, points, greedy| -> Result<()> {
        match example.render_mode {
            RenderMode::CompletedAssistantTranscript => {
                let first = points.first().context("completed transcript has no row")?;
                let last = points.last().unwrap().prefix_token_count;
                let mut point_index = 0usize;
                for prefix in first.prefix_token_count..=last {
                    let (position, input) = if prefix == first.prefix_token_count {
                        (0, &tokens[..prefix])
                    } else {
                        (u32::try_from(prefix - 1)?, &tokens[prefix - 1..prefix])
                    };
                    let emit = points
                        .get(point_index)
                        .is_some_and(|point| point.prefix_token_count == prefix);
                    schedule.record(&example.stable_id, position, input, emit)?;
                    point_index += usize::from(emit);
                }
                ensure!(point_index == points.len() && greedy.is_none());
            }
            RenderMode::GenerationPrompt => {
                schedule.record(&example.stable_id, 0, tokens, true)?;
                let trajectory = target
                    .greedy_trajectories
                    .get(trajectory_index)
                    .context("completed target lacks greedy trajectory")?;
                ensure!(trajectory.stable_id == example.stable_id);
                for (offset, token) in trajectory.token_ids.iter().take(31).enumerate() {
                    schedule.record(
                        &example.stable_id,
                        u32::try_from(tokens.len() + offset)?,
                        std::slice::from_ref(token),
                        true,
                    )?;
                }
                trajectory_index += 1;
            }
        }
        Ok(())
    })?;
    ensure!(trajectory_index == target.greedy_trajectories.len());
    Ok(schedule.finish())
}
