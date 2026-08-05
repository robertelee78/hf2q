//! Ordered command-buffer submission for the one-token verifier.

use anyhow::{Context, Result};
use mlx_native::graph::GraphSession;
use mlx_native::CommandEncoder;

pub(super) type SubmissionChain = Vec<(String, CommandEncoder)>;

pub(super) fn finish_or_commit(
    session: GraphSession<'_>,
    in_flight: Option<&mut SubmissionChain>,
    label: String,
) -> Result<()> {
    if let Some(in_flight) = in_flight {
        in_flight.push((label, session.commit()));
        Ok(())
    } else {
        session.finish().with_context(|| label)
    }
}

pub(super) fn drain(in_flight: &SubmissionChain) -> Result<()> {
    let terminal_result = in_flight
        .last()
        .map(|(_, last)| last.wait_until_completed());
    let profile_stages = std::env::var("HF2Q_DEEPSEEK_STAGE_PROFILE").as_deref() == Ok("1");
    let mut first_error = None;
    for (index, (label, encoder)) in in_flight.iter().enumerate() {
        match encoder.wait_until_completed() {
            Ok(()) => {
                encoder.accumulate_gpu_busy();
                if profile_stages {
                    mlx_native::kernel_profile::record(label, encoder.completed_gpu_interval_ns());
                }
            }
            Err(error) if first_error.is_none() => {
                first_error = Some(anyhow::Error::new(error).context(format!(
                    "DeepSeek-V4 command buffer {index} ({label}) failed"
                )));
            }
            Err(_) => {}
        }
    }
    if let Some(error) = first_error {
        return Err(error);
    }
    if let Some(result) = terminal_result {
        result.context("DeepSeek-V4 terminal command buffer failed")?;
    }
    Ok(())
}

pub(super) fn retained_reference_pipeline_enabled() -> bool {
    // The pipeline relies on regular Metal command buffers retaining transient
    // resources until completion. The opt-in unretained mode requires a
    // persistent arena and therefore keeps the original synchronous behavior.
    std::env::var_os("MLX_UNRETAINED_REFS").is_none()
}
