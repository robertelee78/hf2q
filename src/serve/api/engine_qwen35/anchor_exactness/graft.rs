//! Real bound-bank identity replacement; same real target weights, never a
//! second full-model runtime or a fabricated graft tensor.
use super::*;
use super::lifecycle::Lane;
pub(super) fn changed_identity(fixture: &mut Fixture) -> Result<()> {
    if fixture.packet.graft.is_none() { return Ok(()); }
    let changed_path = fixture.packet.changed_graft.clone()
        .context("selected graft matrix requires the real changed-identity graft artifact")?;
    ensure!(changed_path.is_absolute() && changed_path.is_file(), "changed graft artifact unavailable");
    let mut lane = Lane::new(fixture)?;
    let ids = super::lifecycle::build(fixture, &mut lane, 3)?;
    let original = lane.store.anchor(ids[1]).context("original graft anchor")?.clone();
    let authority = super::lifecycle::script(&mut lane, &ids)?;
    let authority_sha256 = super::case_evidence::authority(fixture, "graft-identity-change", &authority)?;
    let old_identity = fixture.qwen().model.kv_graft.as_ref().context("original real bound graft")?.identity_hash;
    let gguf = acquisition::open_model(&fixture.packet.model)?;
    let (bank, _) = crate::inference::graft::validate_graft_for_model(
        &changed_path, &fixture.packet.model, &gguf)?;
    let bound = crate::inference::graft::BoundGraft::bind(bank);
    ensure!(bound.identity_hash != old_identity, "changed graft has the same actual activation identity");
    let new_identity = bound.identity_hash;
    fixture.qwen().model.kv_graft = Some(bound);
    // Production cold admission splices the newly bound bank. Its cache-side
    // generation must make the old payload ineligible before any restore write.
    lane.cache.reset_for_slot(SLOT)?;
    lane.retained.clear(); lane.spec = None;
    let prompt = fixture.packet.workload.lanes[0][..128].to_vec();
    let served = super::lifecycle::prefill(fixture, &mut lane, &prompt, 0, false)?;
    super::lifecycle::finish_success(fixture, &mut lane, served)?;
    let before = super::live_ram::read(fixture, &lane.cache, SLOT)?;
    ensure!(lane.cache.plan_slot_anchor_restore(SLOT, &original.kv).is_err(),
        "old graft identity was accepted after actual changed-bank splice");
    before.same(&super::live_ram::read(fixture, &lane.cache, SLOT)?, "graft identity Phase A mutated state")?;
    ensure!(restore_committed_anchor(&mut lane.cache, SLOT, &mut lane.store,
        original.id, &fixture.supervisor).is_err(), "production restore accepted changed graft identity");
    before.same(&super::live_ram::read(fixture, &lane.cache, SLOT)?, "production changed-graft refusal mutated state")?;
    lane.cold()?;
    let mut twin = Lane::new(fixture)?;
    let actual = super::lifecycle::compare_final(fixture, &mut lane, &prompt, 0, false,
        "graft-identity-change", "actual")?;
    let expected = super::lifecycle::compare_final(fixture, &mut twin, &prompt, 0, false,
        "graft-identity-change", "twin")?;
    let name = "graft-identity-change";
    super::trace_pair::write_pair(fixture, name, &actual.0, &expected.0)?;
    ensure!(actual == expected && lane.retained == twin.retained,
        "DivergedExactInvariant: changed graft refused/cold twin");
    super::case_evidence::classification(fixture, "graft-identity-change",
        &serde_json::json!({"schema":"hf2q.c2.graft-identity.v1", "old_identity":old_identity,
            "new_identity":new_identity, "old_restore":"RefusedBeforeMutation",
            "cold_twin":"ExactEqual", "actual_bound_bank":changed_path,"prefix_script_sha256":authority_sha256}))?;
    Ok(())
}
