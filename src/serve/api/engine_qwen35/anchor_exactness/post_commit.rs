//! Isolated per-plan injection after a REAL successful GPU commit/wait. This
//! does not poison global Metal state or fabricate a completed native command.
use super::*;
pub(super) fn run(family: Family) -> Result<()> {
    let _gpu = crate::inference::hf2q_gpu_test_lock();
    let mut fixture = Fixture::load(family)?;
    let mut lane = super::lifecycle::Lane::new(&fixture)?;
    let ids = super::lifecycle::build(&mut fixture, &mut lane, 2)?;
    lane.store.restore_fault = Some(AnchorRestoreFault::PostCommit);
    let error = lane.restore(ids[1], &fixture.supervisor).err()
        .context("actual post-commit restore fault did not fire")?;
    ensure!(crate::serve::api::engine::is_fatal_command_buffer_error(&error),
        "committed CommandBufferError lost its typed fatal classifier (or selected restore route is not GPU)");
    ensure!(lane.store.is_poisoned() && lane.store.open_transaction().is_none(),
        "post-commit failure left a live ticket/store");
    // No cold replay or shallower fallback after this typed error.
    fs::write(fixture.packet.output.join("post-commit-fatal.json"), serde_json::to_vec_pretty(
        &serde_json::json!({"schema":"hf2q.c2.post-commit-fatal.v1",
            "actual_gpu_commit_before_injection":true,"typed_fatal_classifier":true,
            "cold_fallback_attempts":0,"global_metal_fault_injection":false}))?)?;
    fixture.finish()
}
