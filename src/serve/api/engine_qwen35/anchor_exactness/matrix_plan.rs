//! A matrix shard selects complete source-defined cases, not tensor subsets.
//! All grants and the full catalog identity are frozen before model loading.
use super::*;
use std::collections::BTreeSet;

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub(super) enum Mode { Lifecycle, O1 }

pub(super) fn catalog(mode: Mode, graft: bool) -> Vec<String> {
    match mode {
        Mode::Lifecycle => {
            let mut cases = Vec::with_capacity(15);
            for rank in [0, 2] { for equality in [false, true] { for peers in [false, true] {
                cases.push(format!("rank{rank}-eq{equality}-peers{peers}"));
            }}}
            cases.extend(["failure-Preflight", "failure-Encode", "cancel-beforecapture",
                "cancel-pending", "cancel-resumed-decode", "cancel-fresh-reset"]
                .into_iter().map(str::to_owned));
            if graft { cases.push("graft-identity-change".into()); }
            cases
        }
        Mode::O1 => {
            let mut cases = Vec::with_capacity(4);
            for shallower in [false, true] { for disabled in [false, true] {
                cases.push(format!("o1-shallower{shallower}-pruningdisabled{disabled}"));
            }}
            cases
        }
    }
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Grants {
    pub events: usize,
    pub child_stores: u64,
    pub records: u64,
    pub payload_bytes: u64,
    pub metadata_bytes: u64,
    pub retained_raw_bytes: u64,
    pub state_readback_bytes: u64,
    pub live_ram_bytes: u64,
}

impl Grants {
    fn validate(&self) -> Result<()> {
        ensure!((1..=65536).contains(&self.events) && self.child_stores > 0 && self.records > 0
            && self.payload_bytes > 0 && self.metadata_bytes > 0
            && self.retained_raw_bytes > 0 && self.state_readback_bytes > 0 && self.live_ram_bytes > 0
            && self.retained_raw_bytes <= self.live_ram_bytes && self.state_readback_bytes <= self.live_ram_bytes,
            "finite nonzero aggregate C2 grants required");
        let reserved = self.events.checked_mul(std::mem::size_of::<super::obligations::Entry>())
            .context("C2 obligation allocation grant overflow")?;
        ensure!(u64::try_from(reserved)? <= self.metadata_bytes,
            "C2 metadata grant does not cover its reserved obligation ledger");
        Ok(())
    }
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Manifest {
    pub schema: String,
    pub identity: Identity,
    pub mode: Mode,
    pub catalog_sha256: String,
    pub cases: Vec<String>,
    pub grants: Grants,
}

impl Manifest {
    pub(super) fn load(binding: &acquisition::FileBinding, identity: &Identity,
        mode: Mode, graft: bool) -> Result<Self> {
        let raw = acquisition::checked_file(binding, 1024 * 1024)?;
        let manifest: Self = serde_json::from_slice(&raw)?;
        manifest.validate(identity, mode, graft)?;
        Ok(manifest)
    }

    pub(super) fn validate(&self, identity: &Identity, mode: Mode, graft: bool) -> Result<()> {
        ensure!(self.schema == "hf2q.c2.matrix-registration.v1" && self.identity == *identity
            && self.mode == mode, "C2 matrix manifest source/identity/mode differs");
        let full = catalog(mode, graft);
        ensure!(self.catalog_sha256 == hash(&serde_json::to_vec(&full)?),
            "C2 manifest does not bind the complete original case catalog");
        let selected: BTreeSet<_> = self.cases.iter().collect();
        ensure!(!selected.is_empty() && selected.len() == self.cases.len()
            && self.cases.iter().all(|case| full.contains(case)),
            "C2 shard must select unique complete original cases");
        self.grants.validate()
    }

    pub(super) fn selected(&self, name: &str) -> bool {
        self.cases.iter().any(|case| case == name)
    }
}

#[derive(Default, Debug, Serialize)]
pub(super) struct Spent {
    pub child_stores: u64,
    pub records: u64,
    pub payload_bytes: u64,
    pub metadata_bytes: u64,
    pub retained_raw_bytes: u64,
    pub live_ram_bytes: u64,
    pub peak_live_ram_bytes: u64,
    pub preflight_case_peak_bytes: u64,
}

#[derive(Serialize)]
pub(super) struct Artifact {
    case_index: usize,
    binding: Option<acquisition::FileBinding>,
}

pub(super) fn required_artifacts(case: &str) -> Vec<String> {
    let mut keys = vec![format!("{case}-authority"), format!("{case}-classification")];
    if case.starts_with("cancel-") {
        for arm in ["actual", "twin"] {
            for suffix in ["-state", "-transactions", "-execution"] { keys.push(format!("{case}-{arm}{suffix}")); }
            if super::request_kind::Purpose::for_case(case, arm).expect("compiled cancellation catalog").raw_required() {
                keys.push(format!("{case}-{arm}"));
            }
        }
        return keys;
    }
    let peers = case.ends_with("-peerstrue");
    let arms: &[&str] = if peers { &["actual", "twin", "single"] } else { &["actual", "twin"] };
    keys.reserve(arms.len() * 4 + usize::from(peers) * 2);
    for arm in arms {
        for suffix in ["", "-state", "-transactions", "-execution"] {
            keys.push(format!("{case}-{arm}{suffix}"));
        }
    }
    if peers {
        for arm in ["actual", "twin"] { keys.push(format!("{case}-single-slot-parity-{arm}")); }
    }
    keys
}

pub(super) struct Budget {
    pub manifest: Manifest,
    pub spent: Spent,
    failed: bool,
    completed: BTreeSet<String>,
    artifacts: std::collections::BTreeMap<String, Artifact>,
    pending_artifacts: Vec<usize>,
}

impl Budget {
    pub(super) fn new(manifest: Manifest) -> Self {
        let mut artifacts = std::collections::BTreeMap::new();
        let mut pending_artifacts = Vec::with_capacity(manifest.cases.len());
        for (case_index, case) in manifest.cases.iter().enumerate() {
            let required = required_artifacts(case);
            pending_artifacts.push(required.len());
            for key in required { artifacts.insert(key, Artifact { case_index, binding: None }); }
        }
        Self { manifest, spent: Spent::default(), failed: false, completed: BTreeSet::new(),
            artifacts, pending_artifacts }
    }
    fn checked<T>(&mut self, operation: impl FnOnce(&mut Self) -> Result<T>) -> Result<T> {
        ensure!(!self.failed, "C2 acquisition budget previously failed");
        let result = operation(self);
        if result.is_err() { self.failed = true; }
        result
    }
    fn reserve(&mut self, records: usize, payload: u64, child_count: u64) -> Result<()> {
        self.checked(|this| {
            ensure!(records <= 65536 && (records == 0) == (payload == 0),
                "invalid exact Q1 reservation");
            let children = this.spent.child_stores.checked_add(child_count).context("child grant overflow")?;
            let records = this.spent.records.checked_add(u64::try_from(records)?)
                .context("record grant overflow")?;
            let bytes = this.spent.payload_bytes.checked_add(payload).context("payload grant overflow")?;
            ensure!(children <= this.manifest.grants.child_stores
                && records <= this.manifest.grants.records && bytes <= this.manifest.grants.payload_bytes,
                "C2 aggregate store/record/disk payload grant exhausted");
            this.spent.child_stores = children;
            this.spent.records = records;
            this.spent.payload_bytes = bytes;
            Ok(())
        })
    }
    pub(super) fn reserve_store(&mut self, records: usize, payload: u64) -> Result<()> {
        self.reserve(records, payload, 1)
    }
    pub(super) fn reserve_child(&mut self) -> Result<()> { self.reserve(0, 0, 1) }
    pub(super) fn reserve_records(&mut self, records: usize, payload: u64) -> Result<()> {
        self.reserve(records, payload, 0)
    }
    pub(super) fn invalidate(&mut self) { self.failed = true; }
    pub(super) fn reserve_metadata(&mut self, bytes: u64) -> Result<()> {
        self.checked(|this| {
            let total = this.spent.metadata_bytes.checked_add(bytes).context("metadata grant overflow")?;
            ensure!(total <= this.manifest.grants.metadata_bytes, "C2 metadata grant exhausted");
            this.spent.metadata_bytes = total;
            Ok(())
        })
    }
    pub(super) fn retain_raw(&mut self, bytes: u64) -> Result<()> {
        self.checked(|this| {
            let total = this.spent.retained_raw_bytes.checked_add(bytes).context("retained raw grant overflow")?;
            ensure!(total <= this.manifest.grants.retained_raw_bytes, "C2 retained raw RAM grant exhausted");
            this.reserve_live(bytes)?;
            this.spent.retained_raw_bytes = total;
            Ok(())
        })
    }
    pub(super) fn state_readback(&mut self, bytes: u64) -> Result<()> {
        self.checked(|this| {
            ensure!(bytes <= this.manifest.grants.state_readback_bytes,
                "C2 complete semantic-state readback exceeds its RAM grant");
            Ok(())
        })
    }
    pub(super) fn preflight_live(&mut self, bytes: u64) -> Result<()> {
        self.checked(|this| {
            ensure!(bytes <= this.manifest.grants.live_ram_bytes,
                "complete-case overlapping cache/anchor/state/raw RAM envelope exceeds admission");
            this.spent.preflight_case_peak_bytes = bytes;
            Ok(())
        })
    }
    pub(super) fn reserve_live(&mut self, bytes: u64) -> Result<()> {
        self.checked(|this| {
            let total = this.spent.live_ram_bytes.checked_add(bytes).context("live RAM grant overflow")?;
            ensure!(bytes > 0 && total <= this.manifest.grants.live_ram_bytes,
                "overlapping C2 live allocation RAM grant exhausted");
            this.spent.live_ram_bytes = total;
            this.spent.peak_live_ram_bytes = this.spent.peak_live_ram_bytes.max(total);
            Ok(())
        })
    }
    pub(super) fn release_live(&mut self, bytes: u64) {
        match self.spent.live_ram_bytes.checked_sub(bytes) {
            Some(remaining) => self.spent.live_ram_bytes = remaining,
            None => { self.failed = true; self.spent.live_ram_bytes = 0; }
        }
    }
    pub(super) fn release_raw(&mut self, bytes: u64) {
        match self.spent.retained_raw_bytes.checked_sub(bytes) {
            Some(remaining) => self.spent.retained_raw_bytes = remaining,
            None => { self.failed = true; self.spent.retained_raw_bytes = 0; }
        }
        self.release_live(bytes);
    }
    pub(super) fn artifacts(&self) -> &std::collections::BTreeMap<String, Artifact> { &self.artifacts }
    pub(super) fn artifact(&self, key: &str) -> Result<&acquisition::FileBinding> {
        self.artifacts.get(key).and_then(|artifact| artifact.binding.as_ref())
            .context("required acquisition artifact has not completed")
    }
    pub(super) fn record_artifact(&mut self, case: &str, key: &str, scope_root: &Path,
        path: &Path, sha256: String) -> Result<()> {
        self.checked(|this| {
            let artifact = this.artifacts.get_mut(key)
                .context("artifact is not in the independently required complete-case set")?;
            ensure!(this.manifest.cases[artifact.case_index] == case && artifact.binding.is_none(),
                "artifact belongs to another case or completed twice");
            ensure!(sha256.len() == 64 && sha256.bytes()
                .all(|c| c.is_ascii_digit() || (b'a'..=b'f').contains(&c)),
                "invalid bound artifact digest");
            ensure!(path.is_file() && path.canonicalize()? == path,
                "bound artifact must be an actual canonical file");
            let relative = path.strip_prefix(scope_root).context("artifact escaped acquisition scope")?;
            ensure!(!relative.as_os_str().is_empty()
                && relative.components().all(|part| matches!(part, std::path::Component::Normal(_))),
                "invalid relative artifact identity");
            artifact.binding = Some(acquisition::FileBinding { path: relative.into(), sha256 });
            this.pending_artifacts[artifact.case_index] = this.pending_artifacts[artifact.case_index]
                .checked_sub(1).context("artifact completion count underflow")?;
            Ok(())
        })
    }

    /// Called only after all arm raw payloads have been dropped.
    pub(super) fn completed_case(&mut self, name: &str) -> Result<()> {
        self.checked(|this| {
            ensure!(this.manifest.selected(name) && this.completed.insert(name.into()),
                "C2 case was undeclared or completed twice");
            let index = this.manifest.cases.iter().position(|case| case == name)
                .context("completed case was not selected")?;
            ensure!(this.pending_artifacts[index] == 0,
                "Incomplete: a complete case omitted independently required store/execution evidence");
            this.release_live(this.spent.retained_raw_bytes);
            this.spent.retained_raw_bytes = 0;
            Ok(())
        })
    }
    pub(super) fn finish(&mut self) -> Result<()> {
        self.checked(|this| {
            ensure!(this.completed.len() == this.manifest.cases.len()
                && this.manifest.cases.iter().all(|case| this.completed.contains(case)),
                "Incomplete: C2 shard omitted a declared complete case");
            Ok(())
        })
    }
}
