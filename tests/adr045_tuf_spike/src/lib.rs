//! Test-only comparator for the ADR-045 signed-update verifier spike.

#[cfg(test)]
mod application_binding;
#[cfg(test)]
mod candidates;
#[cfg(test)]
mod capture_transport;
#[cfg(test)]
#[allow(dead_code)]
#[path = "../../../src/distribution/schema/common.rs"]
mod common;
#[cfg(test)]
mod generation_journal;
#[cfg(test)]
mod model;
#[cfg(test)]
#[allow(dead_code)]
#[path = "../../../src/distribution/schema/release_manifest.rs"]
mod production_release_manifest;
#[cfg(test)]
mod strict_json;
#[cfg(test)]
mod test_repository;
#[cfg(test)]
mod tests;
