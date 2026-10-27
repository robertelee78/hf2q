//! Runtime joins to independently reviewed build/artifact evidence. A packet
//! approval boolean is never a replacement for the separate review receipt.
use super::{protocol::Workload, receipt::*, store::hash};
use anyhow::{ensure, Context, Result};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{
    cell::RefCell,
    fs::File,
    io::Read,
    os::unix::fs::MetadataExt,
    path::{Path, PathBuf},
};
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FileIdentity {
    pub device: u64,
    pub inode: u64,
    pub bytes: u64,
    pub mtime: i64,
    pub mtime_ns: i64,
    pub ctime: i64,
    pub ctime_ns: i64,
}
impl FileIdentity {
    pub fn observed(file: &File) -> Result<Self> {
        let m = file.metadata()?;
        ensure!(m.is_file(), "bound model is not a regular file");
        Ok(Self {
            device: m.dev(),
            inode: m.ino(),
            bytes: m.len(),
            mtime: m.mtime(),
            mtime_ns: m.mtime_nsec(),
            ctime: m.ctime(),
            ctime_ns: m.ctime_nsec(),
        })
    }
}
fn same_opened_file(expected: &FileIdentity, observed: &FileIdentity) -> Result<()> {
    ensure!(expected == observed, "opened model FD identity changed");
    Ok(())
}
#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FileBinding {
    pub path: PathBuf,
    pub sha256: String,
}
#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Binding {
    pub composition: FileBinding,
    pub independent_review: FileBinding,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Composition {
    schema: String,
    comparison_key: ComparisonKey,
    model_file: FileIdentity,
    full_content_receipt: FileBinding,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Review {
    schema: String,
    verdict: String,
    composition_sha256: String,
}
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ComparisonKey {
    pub family: Family,
    pub source_commit: String,
    pub binary_sha256: String,
    pub native_version: String,
    pub native_checksum: String,
    pub model_sha256: String,
    pub tokenizer_sha256: String,
    pub template_sha256: String,
    pub config_sha256: String,
    pub token_vectors_sha256: String,
}
impl From<&Identity> for ComparisonKey {
    fn from(i: &Identity) -> Self {
        Self {
            family: i.family,
            source_commit: i.source_commit.clone(),
            binary_sha256: i.binary_sha256.clone(),
            native_version: i.native_version.clone(),
            native_checksum: i.native_checksum.clone(),
            model_sha256: i.model_sha256.clone(),
            tokenizer_sha256: i.tokenizer_sha256.clone(),
            template_sha256: i.template_sha256.clone(),
            config_sha256: i.config_sha256.clone(),
            token_vectors_sha256: i.token_vectors_sha256.clone(),
        }
    }
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Acquisition {
    pub packet_sha256: String,
    pub nonce: String,
    pub output_root: PathBuf,
    pub actual_binary_sha256: String,
    pub composition_sha256: String,
    pub independent_review_sha256: String,
    pub opened_model: FileIdentity,
    pub observed_model_opens: u64,
    pub required_model_opens: u64,
    pub device_registry_id: u64,
    pub queue_identity: u64,
    pub model_instance: u64,
    pub provenance: String,
}
#[derive(Clone, Debug, Serialize)]
pub struct ConfigRecipe<'a> {
    pub family: Family,
    pub capacity: u64,
    pub slots: u32,
    pub output_allowance: u64,
    pub cache_mode: &'a str,
    pub route: &'a str,
    pub actual_model_config_sha256: String,
}
pub fn tokens_digest(work: &Workload) -> Result<String> {
    work.validate().map_err(anyhow::Error::msg)?;
    let mut digest = Sha256::new();
    digest.update(b"hf2q.q1.tokens.le-u32.v1\0");
    digest.update((work.lanes.len() as u32).to_le_bytes());
    for lane in &work.lanes {
        digest.update((lane.len() as u32).to_le_bytes());
        for token in lane {
            digest.update(token.to_le_bytes());
        }
    }
    Ok(hex::encode(digest.finalize()))
}
pub fn config_digest(recipe: &ConfigRecipe<'_>) -> Result<String> {
    Ok(hash(&serde_json::to_vec(recipe)?))
}
fn checked_file(binding: &FileBinding, max: u64) -> Result<Vec<u8>> {
    ensure!(
        binding.sha256.len() == 64 && binding.path.is_absolute(),
        "invalid external proof binding"
    );
    let meta = std::fs::symlink_metadata(&binding.path)?;
    ensure!(
        meta.is_file() && !meta.file_type().is_symlink() && meta.len() <= max,
        "external proof extent/type"
    );
    let mut bytes = Vec::new();
    File::open(&binding.path)?
        .take(max + 1)
        .read_to_end(&mut bytes)?;
    ensure!(
        bytes.len() as u64 <= max && hash(&bytes) == binding.sha256,
        "external proof hash differs"
    );
    Ok(bytes)
}
fn full_content_receipt_has_model(bytes: &[u8], model_sha: &str) -> Result<()> {
    fn has(value: &serde_json::Value, sha: &str) -> bool {
        match value {
            serde_json::Value::String(s) => s == sha,
            serde_json::Value::Array(values) => values.iter().any(|v| has(v, sha)),
            serde_json::Value::Object(values) => values.values().any(|v| has(v, sha)),
            _ => false,
        }
    }
    let value: serde_json::Value = serde_json::from_slice(bytes)?;
    ensure!(
        has(&value, model_sha),
        "retained full-content receipt does not name model digest"
    );
    Ok(())
}
fn executable_digest() -> Result<String> {
    let mut file = File::open(std::env::current_exe()?)?;
    let mut digest = Sha256::new();
    let mut page = [0u8; 65536];
    loop {
        let n = file.read(&mut page)?;
        if n == 0 {
            break;
        }
        digest.update(&page[..n]);
    }
    Ok(hex::encode(digest.finalize()))
}
struct Active {
    file: File,
    expected: FileIdentity,
    opens: u64,
    nonce: String,
}
thread_local! {static ACTIVE:RefCell<Option<Active>>=const{RefCell::new(None)};}
pub struct Scope {
    acquisition: Acquisition,
    config_checked: std::cell::Cell<bool>,
    _same_thread: std::marker::PhantomData<std::rc::Rc<()>>,
}
impl Scope {
    /// Before any model load/GPU work: verify actual opened FD and executable
    /// against a separate root-reviewed composition, not a packet assertion.
    pub fn begin(
        binding: &Binding,
        identity: &Identity,
        work: &Workload,
        model: &Path,
        output: &Path,
        packet: &[u8],
    ) -> Result<Self> {
        identity.validate().map_err(anyhow::Error::msg)?;
        ensure!(
            identity.family == work.family && tokens_digest(work)? == identity.token_vectors_sha256,
            "actual workload digest/family mismatch"
        );
        let composition: Composition =
            serde_json::from_slice(&checked_file(&binding.composition, 1024 * 1024)?)?;
        let review: Review =
            serde_json::from_slice(&checked_file(&binding.independent_review, 65536)?)?;
        ensure!(
            composition.schema == "hf2q.q1.acquisition-composition.v1"
                && review.schema == "hf2q.q1.acquisition-review.v1"
                && review.verdict == "Accepted"
                && review.composition_sha256 == binding.composition.sha256,
            "independent acquisition composition is unverified"
        );
        ensure!(
            composition.comparison_key == ComparisonKey::from(identity),
            "composition immutable comparison key mismatch"
        );
        let historical_full_content_receipt =
            checked_file(&composition.full_content_receipt, 16 * 1024 * 1024)?;
        full_content_receipt_has_model(&historical_full_content_receipt, &identity.model_sha256)?;
        ensure!(
            option_env!("HF2Q_BUILD_GIT_SHA") == Some(identity.source_commit.as_str()),
            "compiled source stamp absent/wrong"
        );
        let actual_binary_sha256 = executable_digest()?;
        ensure!(
            actual_binary_sha256 == identity.binary_sha256,
            "actual executable differs from reviewed build"
        );
        let file = File::open(model)?;
        let opened = FileIdentity::observed(&file)?;
        same_opened_file(&composition.model_file, &opened)?;
        ensure!(
            output.is_absolute()
                && output.parent().context("output parent")?.canonicalize()?
                    == output.parent().unwrap(),
            "canonical acquisition output parent required"
        );
        let mut nonce = [0u8; 32];
        File::open("/dev/urandom")?.read_exact(&mut nonce)?;
        let nonce = hex::encode(nonce);
        ACTIVE.with(|slot| -> Result<()> {
            ensure!(slot.borrow().is_none(), "nested acquisition");
            *slot.borrow_mut() = Some(Active {
                file,
                expected: opened.clone(),
                opens: 0,
                nonce: nonce.clone(),
            });
            Ok(())
        })?;
        Ok(Self {
            acquisition: Acquisition {
                packet_sha256: hash(packet),
                nonce,
                output_root: output.to_path_buf(),
                actual_binary_sha256,
                composition_sha256: binding.composition.sha256.clone(),
                independent_review_sha256: binding.independent_review.sha256.clone(),
                opened_model: opened,
                observed_model_opens: 0,
                required_model_opens: 0,
                device_registry_id: 0,
                queue_identity: 0,
                model_instance: 0,
                provenance: "IndependentRootReviewedComposition".into(),
            },
            config_checked: std::cell::Cell::new(false),
            _same_thread: std::marker::PhantomData,
        })
    }
    pub fn check_config(&self, identity: &Identity, recipe: &ConfigRecipe<'_>) -> Result<()> {
        ensure!(
            config_digest(recipe)? == identity.config_sha256,
            "actual model/cache/route config differs from immutable comparison key"
        );
        self.config_checked.set(true);
        Ok(())
    }
    pub fn check_unchanged(&self) -> Result<()> {
        ACTIVE.with(|slot| -> Result<()> {
            let slot = slot.borrow();
            let active = slot.as_ref().context("acquisition disappeared")?;
            ensure!(
                active.nonce == self.acquisition.nonce
                    && FileIdentity::observed(&active.file)? == active.expected,
                "model FD mutated during acquisition"
            );
            Ok(())
        })
    }
    pub fn observed(
        &self,
        device: &mlx_native::MlxDevice,
        model_instance: u64,
        required_model_opens: u64,
    ) -> Result<Acquisition> {
        use mlx_native::metal::foreign_types::ForeignType;
        ensure!(
            self.config_checked.get(),
            "actual config recipe was not checked"
        );
        let mut result = self.acquisition.clone();
        ACTIVE.with(|slot| -> Result<()> {
            let slot = slot.borrow();
            let active = slot.as_ref().context("acquisition disappeared")?;
            ensure!(
                active.nonce == result.nonce
                    && FileIdentity::observed(&active.file)? == active.expected
                    && required_model_opens > 0
                    && active.opens >= required_model_opens,
                "model FD changed or actual loader did not consume bound descriptor"
            );
            result.observed_model_opens = active.opens;
            result.required_model_opens = required_model_opens;
            Ok(())
        })?;
        result.device_registry_id = device.registry_id();
        result.queue_identity = device.metal_queue().as_ptr() as usize as u64;
        result.model_instance = model_instance;
        ensure!(
            result.device_registry_id != 0 && result.queue_identity != 0 && model_instance != 0,
            "missing observed runtime identity"
        );
        Ok(result)
    }
}
impl Drop for Scope {
    fn drop(&mut self) {
        ACTIVE.with(|slot| {
            let mut slot = slot.borrow_mut();
            if slot
                .as_ref()
                .is_some_and(|a| a.nonce == self.acquisition.nonce)
            {
                slot.take();
            }
        });
    }
}
/// Test-only loader seam. The exact File checked by fstat is consumed by the
/// production parser; ordinary tests without a scope keep the existing open.
pub fn open_model(path: &Path) -> Result<mlx_native::gguf::GgufFile> {
    let file = File::open(path)?;
    ACTIVE.with(|slot| -> Result<()> {
        if let Some(active) = slot.borrow_mut().as_mut() {
            same_opened_file(&active.expected, &FileIdentity::observed(&file)?)?;
            active.opens = active
                .opens
                .checked_add(1)
                .context("model open count overflow")?;
        }
        Ok(())
    })?;
    Ok(mlx_native::gguf::GgufFile::from_file(file)?)
}

pub fn validate_pair_for(
    left_root: &Path,
    left: &Trace,
    right_root: &Path,
    right: &Trace,
    recovery: bool,
) -> std::result::Result<bool, String> {
    let l = left
        .acquisition
        .as_ref()
        .ok_or("left acquisition unverified")?;
    let r = right
        .acquisition
        .as_ref()
        .ok_or("right acquisition unverified")?;
    if (l.nonce == r.nonce) != recovery {
        return Err("acquisition nonce does not match comparison purpose".into());
    }
    let synthetic =
        l.provenance == "SyntheticOracleControl" && r.provenance == "SyntheticOracleControl";
    if !synthetic && !recovery && l.packet_sha256 == r.packet_sha256 {
        return Err("same acquisition packet cannot be relabelled across trajectories".into());
    }
    for (root, trace, a) in [(left_root, left, l), (right_root, right, r)] {
        let hex = |s: &str| {
            s.len() == 64
                && s.bytes()
                    .all(|b| b.is_ascii_hexdigit() && !b.is_ascii_uppercase())
        };
        if a.output_root != root
            || !hex(&a.nonce)
            || !hex(&a.packet_sha256)
            || a.actual_binary_sha256 != trace.comparison_key.binary_sha256
        {
            return Err("acquisition output/nonce/packet/build binding mismatch".into());
        }
        if !synthetic
            && (a.provenance != "IndependentRootReviewedComposition"
                || !hex(&a.composition_sha256)
                || !hex(&a.independent_review_sha256)
                || (a.required_model_opens == 0 || a.observed_model_opens < a.required_model_opens)
                || a.opened_model.inode == 0
                || a.opened_model.bytes == 0
                || a.device_registry_id == 0
                || a.queue_identity == 0
                || a.model_instance == 0)
        {
            return Err("unverified actual acquisition provenance".into());
        }
    }
    Ok(!synthetic)
}
#[cfg(test)]
pub(super) fn synthetic(root: &Path, identity: &Identity) -> Acquisition {
    Acquisition {
        packet_sha256: hash(b"synthetic-oracle-fixture"),
        nonce: hash(root.as_os_str().as_encoded_bytes()),
        output_root: root.into(),
        actual_binary_sha256: identity.binary_sha256.clone(),
        composition_sha256: String::new(),
        independent_review_sha256: String::new(),
        opened_model: FileIdentity {
            device: 0,
            inode: 0,
            bytes: 0,
            mtime: 0,
            mtime_ns: 0,
            ctime: 0,
            ctime_ns: 0,
        },
        observed_model_opens: 0,
        required_model_opens: 0,
        device_registry_id: 0,
        queue_identity: 0,
        model_instance: 0,
        provenance: "SyntheticOracleControl".into(),
    }
}

#[cfg(test)]
#[path = "acquisition_controls.rs"]
mod controls;
