//! Bounded current borrowed-owner observations, never a physical/creation ledger.
use anyhow::{ensure, Context, Result};
use mlx_native::{MlxBuffer, RootOriginSnapshot};
use serde::Serialize;
use std::collections::BTreeMap;

pub(crate) const MAX_FIELDS: usize = 8192;
pub(crate) const MAX_JSON_BYTES: usize = 16 * 1024 * 1024;
#[derive(Debug, Serialize)]
#[serde(tag = "presence")]
pub(crate) enum Field {
    AbsentByActualOption,
    Present {
        native_root_id: usize,
        requested_parent_bytes: usize,
        view_offset: u64,
        logical_bytes: usize,
        dtype: String,
        shape: Vec<usize>,
    },
}
impl Field {
    pub(crate) fn actual(buffer: Option<&MlxBuffer>) -> Result<Self> {
        let Some(buffer) = buffer else {
            return Ok(Self::AbsentByActualOption);
        };
        ensure!(buffer.shape().len() <= 16, "Q5 role rank cap exceeded");
        let backing = buffer.backing_snapshot()?;
        let RootOriginSnapshot::KnownNative {
            id,
            requested_bytes,
        } = backing.origin
        else {
            anyhow::bail!("Q5 current role native identity Unknown");
        };
        Ok(Self::Present {
            native_root_id: id,
            requested_parent_bytes: requested_bytes,
            view_offset: backing.view_byte_offset,
            logical_bytes: backing.logical_bytes,
            dtype: buffer.dtype().to_string(),
            shape: buffer.shape().to_vec(),
        })
    }
}
#[derive(Debug, Serialize)]
pub(crate) struct CurrentManifest {
    pub sequence: u64,
    pub clock_id: &'static str,
    pub begin_ns: u64,
    pub end_ns: Option<u64>,
    pub gpu_registry_id: u64,
    pub selected_logical_owners: Vec<(u64, u32, u64)>,
    pub lifecycle: Option<&'static str>,
    pub fields: BTreeMap<String, Field>,
    pub cache_states: BTreeMap<String, crate::inference::models::deepseek4::cache::Q5CacheState>,
}
impl CurrentManifest {
    pub(crate) fn new(
        sequence: u64,
        gpu_registry_id: u64,
        owners: &[(u64, u32, u64)],
        lifecycle: Option<&'static str>,
    ) -> Result<Self> {
        ensure!(
            gpu_registry_id != 0 && owners.len() <= 4,
            "Q5 current manifest device/owner bound"
        );
        Ok(Self {
            sequence,
            clock_id: "POSIX_CLOCK_MONOTONIC",
            begin_ns: monotonic_ns()?,
            end_ns: None,
            gpu_registry_id,
            selected_logical_owners: owners.to_vec(),
            lifecycle,
            fields: BTreeMap::new(),
            cache_states: BTreeMap::new(),
        })
    }
    pub(crate) fn field(&mut self, label: &str, buffer: Option<&MlxBuffer>) -> Result<()> {
        ensure!(
            self.end_ns.is_none()
                && !label.is_empty()
                && label.len() <= 256
                && self.fields.len() < MAX_FIELDS,
            "Q5 current role field capacity/name cap"
        );
        ensure!(
            !self.fields.contains_key(label),
            "Q5 duplicate current role"
        );
        self.fields.insert(label.to_owned(), Field::actual(buffer)?);
        Ok(())
    }
    pub(crate) fn close(&mut self) -> Result<()> {
        let end = monotonic_ns()?;
        ensure!(
            self.end_ns.is_none() && end >= self.begin_ns,
            "Q5 manifest clock/close mismatch"
        );
        self.end_ns = Some(end);
        Ok(())
    }
}
pub(crate) fn monotonic_ns() -> Result<u64> {
    let mut time = libc::timespec {
        tv_sec: 0,
        tv_nsec: 0,
    };
    // SAFETY: valid writable stack timespec and fixed POSIX clock ID.
    ensure!(
        unsafe { libc::clock_gettime(libc::CLOCK_MONOTONIC, &mut time) } == 0,
        "Q5 POSIX monotonic clock unavailable"
    );
    ensure!(
        time.tv_sec >= 0 && (0..1_000_000_000).contains(&time.tv_nsec),
        "Q5 invalid monotonic timespec"
    );
    u64::try_from(time.tv_sec)?
        .checked_mul(1_000_000_000)
        .and_then(|v| v.checked_add(time.tv_nsec as u64))
        .context("Q5 clock overflow")
}
