//! Case classification and input authority are required, digest-bound artifacts.
use super::*;

pub(super) fn authority(fixture: &Fixture, case: &str, value: &impl Serialize) -> Result<String> {
    publish(fixture, case, "authority", &format!("{case}-prefix-script.json"), value)
}
pub(super) fn classification(fixture: &Fixture, case: &str, value: &impl Serialize) -> Result<String> {
    publish(fixture, case, "classification", &format!("{case}.json"), value)
}
fn publish(fixture: &Fixture, case: &str, kind: &str, name: &str, value: &impl Serialize) -> Result<String> {
    let budget = fixture.matrix_budget()?;
    let path = fixture.packet.output.join(name);
    let sha256 = super::metadata::write(&path, budget.as_ref(), value)?;
    budget.borrow_mut().record_artifact(case, &format!("{case}-{kind}"), &fixture.packet.output,
        &path, sha256.clone())?;
    Ok(sha256)
}

#[derive(Serialize)]
pub(super) struct PeerRestoreDigest {
    pub slot: u32,
    pub before: String,
    pub after: String,
}

pub(super) fn peer_state_digest(slot: SlotId, state: &trace::State) -> String {
    use sha2::{Digest, Sha256};
    let mut hash = Sha256::new();
    hash.update(b"hf2q.c2.peer-semantic-state.le.v1\0");
    hash.update(slot.0.to_le_bytes());
    hash.update((state.target.len() as u64).to_le_bytes());
    for cursor in &state.target { hash.update(cursor.to_le_bytes()); }
    match state.mtp {
        Some(cursor) => { hash.update([1]); hash.update(cursor.to_le_bytes()); }
        None => hash.update([0]),
    }
    hash.update((state.parity.len() as u64).to_le_bytes());
    for parity in &state.parity { hash.update([u8::from(*parity)]); }
    hash.update((state.regions.len() as u64).to_le_bytes());
    for region in &state.regions {
        hash.update((region.len() as u64).to_le_bytes());
        hash.update(region);
    }
    format!("{:x}", hash.finalize())
}

#[test]
fn c2_peer_digest_preserves_cursor_mtp_parity_payload_region_and_slot_identity() {
    let state = trace::State { target: vec![1], mtp: Some(2), parity: vec![false], regions: vec![vec![1, 2]] };
    let digest = peer_state_digest(SlotId(1), &state);
    let mut changed = state.clone(); changed.target[0] += 1;
    assert_ne!(peer_state_digest(SlotId(1), &changed), digest);
    let mut changed = state.clone(); changed.mtp = None;
    assert_ne!(peer_state_digest(SlotId(1), &changed), digest);
    let mut changed = state.clone(); changed.parity[0] = true;
    assert_ne!(peer_state_digest(SlotId(1), &changed), digest);
    let mut changed = state.clone(); changed.regions[0][0] = 9;
    assert_ne!(peer_state_digest(SlotId(1), &changed), digest);
    let mut changed = state.clone(); changed.regions = vec![vec![1], vec![2]];
    assert_ne!(peer_state_digest(SlotId(1), &changed), digest);
    assert_ne!(peer_state_digest(SlotId(2), &state), digest);
}
