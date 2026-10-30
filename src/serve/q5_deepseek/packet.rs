use anyhow::{ensure, Result};
use serde::Deserialize;
use std::{fs::Metadata, os::unix::fs::MetadataExt};

#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Eq)]
#[cfg_attr(test, derive(serde::Serialize))]
#[serde(rename_all = "snake_case")]
pub(super) enum Phase {
    Preload,
    Load,
    Provision,
    Warmup,
    Forward,
    Drain,
    Shutdown,
}

/// A declared future cell cannot satisfy any observed-phase requirement.
#[derive(Debug, Deserialize)]
#[cfg_attr(test, derive(serde::Serialize))]
#[serde(tag = "state", deny_unknown_fields)]
pub(super) enum Cell {
    Planned { evidence_sha256: String },
    Unknown { reason: String },
}

#[derive(Debug, Clone, Copy, Deserialize, PartialEq, Eq)]
#[cfg_attr(test, derive(serde::Serialize))]
#[serde(deny_unknown_fields)]
pub(super) struct FileIdentity {
    pub device: u64,
    pub inode: u64,
    pub bytes: u64,
    pub mtime_seconds: i64,
    pub mtime_nanoseconds: i64,
    pub ctime_seconds: i64,
    pub ctime_nanoseconds: i64,
}
impl FileIdentity {
    pub fn matches(&self, m: &Metadata) -> Result<()> {
        ensure!(
            self.device == m.dev()
                && self.inode == m.ino()
                && self.bytes == m.len()
                && self.mtime_seconds == m.mtime()
                && self.mtime_nanoseconds == m.mtime_nsec()
                && self.ctime_seconds == m.ctime()
                && self.ctime_nanoseconds == m.ctime_nsec(),
            "Q5 actual readonly FD identity changed"
        );
        Ok(())
    }
}

#[derive(Debug, Deserialize)]
#[cfg_attr(test, derive(serde::Serialize))]
#[serde(deny_unknown_fields)]
pub(super) struct Packet {
    pub schema: String,
    #[serde(default)]
    pub operating_case: Option<super::reviewed_budget::Case>,
    pub hf_source_sha256: String,
    pub native_package_sha256: String,
    pub binary_sha256: String,
    pub artifact_sha256: String,
    pub workload_sha256: String,
    pub policy_sha256: String,
    pub file: FileIdentity,
    pub context: u32,
    pub output_limit: u32,
    pub logical_slots: u32,
    pub weight_policy: String,
    pub collector_upper_bytes: u64,
    pub phase: Phase,
    #[serde(deserialize_with = "closed_cells")]
    pub cells: std::collections::BTreeMap<String, Cell>,
}
impl Packet {
    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.schema == "hf2q.q5.deepseek.selected-case.preload.v1",
            "Q5 packet schema mismatch"
        );
        for hash in [
            &self.hf_source_sha256,
            &self.native_package_sha256,
            &self.binary_sha256,
            &self.artifact_sha256,
            &self.workload_sha256,
            &self.policy_sha256,
        ] {
            ensure!(
                hash.len() == 64
                    && hash
                        .bytes()
                        .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b)),
                "Q5 identity requires canonical SHA256"
            );
        }
        ensure!(
            self.context == 262_144 && self.output_limit == 8192 && self.logical_slots == 4,
            "Q5 selected profile changed"
        );
        ensure!(
            self.weight_policy == "FullCoalescedNoCopy",
            "Q5 weight mechanism not admitted"
        );
        ensure!(
            self.collector_upper_bytes > 0,
            "Q5 collector overhead must be charged"
        );
        ensure!(
            self.phase == Phase::Preload,
            "Q5 runtime phase has no published observer permit"
        );
        let required = self.requirements(self.phase)?;
        ensure!(
            self.cells.len() == required.len(),
            "Q5 phase has missing/extra cells"
        );
        for key in required {
            match self.cells.get(*key) {
                Some(Cell::Planned { evidence_sha256 }) => ensure!(
                    evidence_sha256.len() == 64
                        && evidence_sha256
                            .bytes()
                            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b)),
                    "Q5 planned evidence hash malformed"
                ),
                Some(Cell::Unknown { reason }) => ensure!(
                    !reason.is_empty() && reason.len() <= 256,
                    "Q5 Unknown reason invalid"
                ),
                None => anyhow::bail!("Q5 missing phase cell {key}"),
            }
        }
        Ok(())
    }
    pub fn requirements(&self, phase: Phase) -> Result<&'static [&'static str]> {
        // Future required-key sets are source obligations, NOT observed cells or
        // eligibility. validate() still rejects every non-preload phase.
        Ok(match phase {
            Phase::Preload => &[
                "same_fd_header",
                "source_native_binary",
                "weight_requested_capacity",
                "bootstrap_plus_four_sessions",
                "transient_and_pending_upper",
                "private_pool_domain_partition",
                "cpu_os_reaction_upper",
                "fresh_operating_review",
            ],
            Phase::Load => &[
                "same_fd_actual_weight_roots",
                "expanded_f32_roots",
                "loader_cpu_peak",
                "native_early_creation_and_live_coverage",
                "load_batch_reservation",
                "page_observer",
            ],
            Phase::Provision => &[
                "bootstrap_displaced_owner",
                "four_retained_sessions",
                "old_new_cache_overlap",
                "current_committed_pending_anchors",
                "native_live_coverage",
            ],
            Phase::Warmup => &[
                "actual_bootstrap_loaded_cache",
                "four_retained_idle_sessions",
                "real_warmup_every_own_result",
                "no_admitted_or_queued_work",
                "native_whole_live_pending",
                "fresh_os_reaction",
                "same_fd_identity",
            ],
            Phase::Forward => &[
                "actual_request_generation_membership",
                "mounted_owner",
                "three_private_tls_domains",
                "all_local_arenas",
                "submission_inputs",
                "actual_pending_commands_and_status",
                "cpu_request_capacities",
                "native_live_residual",
                "maximum_unobserved_batch",
                "fresh_os_reaction",
                "page_observer",
            ],
            Phase::Drain => &[
                "every_own_checked_result",
                "moe_status_validation",
                "completed_retained_resources",
                "cache_anchor_logit_live_roles",
                "native_live_residual",
                "fresh_os_reaction",
            ],
            Phase::Shutdown => &[
                "all_worker_operations_joined",
                "all_results_and_callbacks_closed",
                "same_fd_final_stat",
                "owned_process_groups_empty",
                "os_recovery",
                "final_metadata_caps",
            ],
        })
    }
}

fn closed_cells<'de, D: serde::Deserializer<'de>>(
    deserializer: D,
) -> Result<std::collections::BTreeMap<String, Cell>, D::Error> {
    struct Cells;
    impl<'de> serde::de::Visitor<'de> for Cells {
        type Value = std::collections::BTreeMap<String, Cell>;
        fn expecting(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            f.write_str("at most eight unique preload required cells")
        }
        fn visit_map<M: serde::de::MapAccess<'de>>(
            self,
            mut map: M,
        ) -> Result<Self::Value, M::Error> {
            let mut cells = std::collections::BTreeMap::new();
            while let Some((key, value)) = map.next_entry::<String, Cell>()? {
                if cells.len() >= 8 || cells.insert(key, value).is_some() {
                    return Err(serde::de::Error::custom(
                        "duplicate/excess Q5 required cells",
                    ));
                }
            }
            Ok(cells)
        }
    }
    deserializer.deserialize_map(Cells)
}
