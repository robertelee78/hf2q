//! Bounded full-logical-extent capture. No GPU synchronization is invented here.
use super::receipt::*;
use sha2::{Digest, Sha256};
use std::{
    collections::BTreeSet,
    fs::{self, OpenOptions},
    io::Write,
    os::unix::fs::{OpenOptionsExt, PermissionsExt},
    path::{Path, PathBuf},
};
pub const PAGE_BYTES: usize = 1024 * 1024;
pub const METADATA_BYTES: u64 = 128 * 1024 * 1024;
// Stream JSON directly; this quota is separate from captured tensor payload.
struct MetadataWriter<W> {
    inner: W,
    used: u64,
}
impl<W: Write> Write for MetadataWriter<W> {
    fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
        let next = self
            .used
            .checked_add(bytes.len() as u64)
            .filter(|&n| n <= METADATA_BYTES)
            .ok_or_else(|| std::io::Error::other("Q1 metadata quota exceeded"))?;
        self.inner.write_all(bytes)?;
        self.used = next;
        Ok(bytes.len())
    }
    fn flush(&mut self) -> std::io::Result<()> {
        self.inner.flush()
    }
}
pub fn hash(bytes: &[u8]) -> String {
    hex::encode(Sha256::digest(bytes))
}
pub struct Store {
    root: PathBuf,
    limit: u64,
    used: u64,
    record_limit: usize,
    trace: Trace,
    seen: BTreeSet<Coordinate>,
}
impl Store {
    pub fn new(
        root: &Path,
        identity: Identity,
        trajectory: String,
        expected: Vec<ExpectedTensor>,
        byte_limit: u64,
        record_limit: usize,
    ) -> Result<Self, String> {
        identity.validate()?;
        if !root.is_absolute()
            || root.exists()
            || byte_limit == 0
            || record_limit == 0
            || trajectory.is_empty()
            || expected.is_empty()
            || expected.len() > record_limit
        {
            return Err("invalid bounded capture packet".into());
        }
        let parent = root.parent().ok_or("capture parent")?;
        if parent.canonicalize().map_err(|e| e.to_string())? != parent
            || parent.ancestors().any(|p| p.join(".git").exists())
        {
            return Err("capture outside canonical private parent required".into());
        }
        let planned = expected
            .iter()
            .try_fold(0u64, |total, e| {
                if e.shape.is_empty()
                    || e.shape.len() > 8
                    || e.shape.contains(&0)
                    || e.at.agent.is_empty()
                    || e.at.agent.len() > 128
                {
                    return None;
                }
                e.shape
                    .iter()
                    .try_fold(e.scalar.bytes() as u64, |n, &d| n.checked_mul(d as u64))
                    .and_then(|n| total.checked_add(n))
            })
            .ok_or("expected diagnostic shape/metadata overflow")?;
        if planned > byte_limit || record_limit > 65536 {
            return Err("expected diagnostic packet exceeds finite quotas".into());
        }
        let unique: BTreeSet<_> = expected.iter().map(|e| e.at.clone()).collect();
        if unique.len() != expected.len() {
            return Err("duplicate expected coordinate".into());
        }
        fs::create_dir(root).map_err(|e| e.to_string())?;
        fs::set_permissions(root, fs::Permissions::from_mode(0o700)).map_err(|e| e.to_string())?;
        Ok(Self {
            root: root.into(),
            limit: byte_limit,
            used: 0,
            record_limit,
            seen: BTreeSet::new(),
            trace: Trace {
                schema: "hf2q.adr063.invariance.v2".into(),
                comparison_key: (&identity).into(),
                identity,
                acquisition: None,
                coverage: Coverage {
                    expected,
                    captured: Vec::new(),
                    errors: Vec::new(),
                    missing_native_capabilities: Vec::new(),
                    trajectory,
                },
            },
        })
    }
    pub fn bind_acquisition(
        &mut self,
        acquisition: super::acquisition::Acquisition,
    ) -> Result<(), String> {
        if self.trace.acquisition.is_some()
            || !self.trace.coverage.captured.is_empty()
            || acquisition.output_root != self.root
            || acquisition.actual_binary_sha256 != self.trace.identity.binary_sha256
        {
            return Err("acquisition must bind exact store before first capture".into());
        }
        self.trace.acquisition = Some(acquisition);
        Ok(())
    }
    pub fn fail(&mut self, why: impl Into<String>) {
        self.trace.coverage.errors.push(why.into());
    }
    pub fn missing_capability(&mut self, why: impl Into<String>) {
        self.trace
            .coverage
            .missing_native_capabilities
            .push(why.into());
    }
    #[allow(clippy::too_many_arguments)]
    pub fn capture(
        &mut self,
        at: Coordinate,
        scalar: Scalar,
        shape: Vec<usize>,
        parent_identity: u64,
        parent_bytes: u64,
        parent_offset: u64,
        submission_id: Option<u64>,
        checked_drain: &str,
        bytes: &[u8],
    ) -> Result<(), String> {
        let result = self.capture_inner(
            at,
            scalar,
            shape,
            parent_identity,
            parent_bytes,
            parent_offset,
            submission_id,
            checked_drain,
            bytes,
        );
        if let Err(error) = &result {
            self.fail(error.clone());
        }
        result
    }
    #[allow(clippy::too_many_arguments)]
    fn capture_inner(
        &mut self,
        at: Coordinate,
        scalar: Scalar,
        shape: Vec<usize>,
        parent_identity: u64,
        parent_bytes: u64,
        parent_offset: u64,
        submission_id: Option<u64>,
        checked_drain: &str,
        bytes: &[u8],
    ) -> Result<(), String> {
        let logical = shape
            .iter()
            .try_fold(1usize, |a, &b| a.checked_mul(b))
            .and_then(|n| n.checked_mul(scalar.bytes()))
            .ok_or("capture extent overflow")?;
        let next = self
            .used
            .checked_add(bytes.len() as u64)
            .ok_or("capture quota overflow")?;
        if shape.is_empty()
            || shape.contains(&0)
            || logical != bytes.len()
            || parent_identity == 0
            || parent_offset
                .checked_add(bytes.len() as u64)
                .is_none_or(|end| end > parent_bytes)
            || checked_drain.trim().is_empty()
            || self.seen.contains(&at)
            || !self
                .trace
                .coverage
                .expected
                .iter()
                .any(|e| e.at == at && e.scalar == scalar && e.shape == shape)
            || next > self.limit
            || self.trace.coverage.captured.len() >= self.record_limit
        {
            return Err("capture extent/identity/drain/coverage/quota refusal".into());
        }
        // Reserve all payload bytes BEFORE create/write. A failed write keeps charge.
        self.used = next;
        let name = format!("tensor-{:06}.bin", self.trace.coverage.captured.len());
        let path = self.root.join(&name);
        let mut file = OpenOptions::new()
            .create_new(true)
            .write(true)
            .mode(0o600)
            .open(path)
            .map_err(|e| e.to_string())?;
        let mut full = Sha256::new();
        let mut pages = Vec::new();
        for (i, page) in bytes.chunks(PAGE_BYTES).enumerate() {
            file.write_all(page).map_err(|e| e.to_string())?;
            full.update(page);
            pages.push(Page {
                offset: (i * PAGE_BYTES) as u64,
                bytes: page.len(),
                sha256: hash(page),
            });
        }
        file.sync_all().map_err(|e| e.to_string())?;
        self.seen.insert(at.clone());
        self.trace.coverage.captured.push(Tensor {
            at,
            scalar,
            shape,
            logical_bytes: bytes.len() as u64,
            parent_identity,
            parent_bytes,
            parent_offset,
            submission_id,
            checked_drain: checked_drain.into(),
            file: name,
            sha256: hex::encode(full.finalize()),
            pages,
        });
        Ok(())
    }
    pub fn finish(mut self) -> Result<Trace, String> {
        if self.seen.len() != self.trace.coverage.expected.len() {
            self.fail("incomplete required logical stage coverage");
        }
        let mut f = OpenOptions::new()
            .create_new(true)
            .write(true)
            .mode(0o600)
            .open(self.root.join("trace.json"))
            .map_err(|e| e.to_string())?;
        serde_json::to_writer(
            &mut MetadataWriter {
                inner: &mut f,
                used: 0,
            },
            &self.trace,
        )
        .map_err(|e| e.to_string())?;
        Ok(self.trace)
    }
}

#[cfg(test)]
mod metadata_controls {
    use super::*;
    #[test]
    fn q1_metadata_writer_denies_before_exceeding_separate_quota() {
        let mut writer = MetadataWriter {
            inner: Vec::new(),
            used: METADATA_BYTES - 2,
        };
        assert!(writer.write_all(b"abc").is_err());
        assert!(writer.inner.is_empty());
        assert_eq!(writer.used, METADATA_BYTES - 2);
        writer.write_all(b"ab").unwrap();
        assert_eq!(writer.used, METADATA_BYTES);
        assert!(writer.write_all(b"c").is_err());
        assert_eq!(writer.inner, b"ab");
    }
}
