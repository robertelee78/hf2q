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
