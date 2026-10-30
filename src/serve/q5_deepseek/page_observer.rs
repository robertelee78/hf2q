//! Public native14 metadata only. Every observation retains FULL requested charge.
//! No flag, absence or alias observation mints physical credit or a load permit.
use super::packet::FileIdentity;
use anyhow::{ensure, Context, Result};
use mlx_native::{
    MlxBuffer, PageObservation, PageObservationSource, PageQueryLimits, PageQueryRange,
};
use serde::Serialize;
use std::fs::File;

pub(super) const MAX_ROOT_QUERIES: usize = 4;
pub(super) const PAGES_PER_ROOT: u32 = 64;
const LIMIT_NS: u64 = 250_000_000;
#[derive(Debug, Serialize)]
pub(super) struct RawQuery {
    clock: &'static str,
    start_ns: u64,
    end_ns: u64,
    page_bytes: usize,
    returned_count: u64,
    dispositions: Vec<i32>,
}
#[derive(Debug, Serialize)]
#[serde(tag = "status")]
pub(super) enum RootObservation {
    UnknownFullCharge {
        reason: &'static str,
    },
    PartialNoCredit {
        root_id: usize,
        requested_bytes: usize,
        same_fd_device: u64,
        same_fd_inode: u64,
        file_offset: u64,
        queried_bytes: usize,
        root: RawQuery,
        ephemeral_alias: RawQuery,
    },
}
fn raw(o: PageObservation) -> RawQuery {
    RawQuery {
        clock: o.clock,
        start_ns: o.start_ns,
        end_ns: o.end_ns,
        page_bytes: o.page_bytes,
        returned_count: o.returned_count,
        dispositions: o.dispositions,
    }
}
/// Pure gate used on ACTUAL native output, before associating it with a file.
fn validate(o: &PageObservation, id: usize, bytes: usize, pages: u32) -> Result<()> {
    ensure!(
        matches!(o.source, PageObservationSource::NoCopyRoot {root_id,requested_bytes,..}
        if root_id==id && requested_bytes==bytes),
        "Q5 page root identity mismatch"
    );
    ensure!(
        o.range.page_offset == 0
            && o.range.page_count == pages
            && o.returned_count == u64::from(pages)
            && o.dispositions.len() == pages as usize
            && o.raw_mach_status == 0
            && o.elapsed_ns <= LIMIT_NS
            && o.end_ns.checked_sub(o.start_ns) == Some(o.elapsed_ns)
            && (4096..=65536).contains(&o.page_bytes)
            && o.page_bytes.is_power_of_two()
            && o.queried_bytes
                == o.page_bytes
                    .checked_mul(pages as usize)
                    .context("Q5 page bytes overflow")?
            && o.dispositions
                .iter()
                .all(|v| *v >= 0 && *v & !mlx_native::KNOWN_PAGE_DISPOSITION_BITS == 0),
        "Q5 page result partial/stale/invalid"
    );
    Ok(())
}
/// Both resource and FD remain borrowed for the entire query. Alias is mapped
/// from that FD, never from an observed raw address; dropped before return.
pub(super) fn actual(
    buffer: &MlxBuffer,
    file: &File,
    identity: FileIdentity,
    registered_file_offset: u64,
    id: usize,
    requested_bytes: usize,
) -> RootObservation {
    let observe = || -> Result<RootObservation> {
        identity.matches(&file.metadata()?)?;
        let b = buffer.backing_snapshot()?;
        let n = b.no_copy.context("Q5 no certified NoCopy")?;
        let range = PageQueryRange {
            page_offset: 0,
            page_count: PAGES_PER_ROOT,
        };
        let limits = PageQueryLimits {
            max_elapsed_ns: LIMIT_NS,
        };
        let root = buffer.observe_nocopy_root_pages(range, limits)?;
        validate(&root, id, requested_bytes, PAGES_PER_ROOT)?;
        ensure!(
            matches!(root.source,PageObservationSource::NoCopyRoot{registered_base,registered_bytes,..}
            if registered_base==n.registered_base_address && registered_bytes==n.registered_bytes),
            "Q5 changed registered range"
        );
        ensure!(
            registered_file_offset % root.page_bytes as u64 == 0
                && registered_file_offset
                    .checked_add(root.queried_bytes as u64)
                    .is_some_and(|end| end <= identity.bytes),
            "Q5 alias escapes verified FD"
        );
        // SAFETY: readonly mapping from retained regular FD; no payload access.
        // Source contract forbids mutation; stat is checked before/after. An
        // external truncation cannot be made safe by this metadata association.
        let alias = unsafe {
            memmap2::MmapOptions::new()
                .offset(registered_file_offset)
                .len(root.queried_bytes)
                .map(file)?
        };
        let observed = mlx_native::observe_readonly_mapping_pages(&alias, range, limits)?;
        ensure!(
            matches!(observed.source,PageObservationSource::ReadonlyMapping {mapping_base,mapping_bytes}
            if mapping_base==alias.as_ptr() as usize && mapping_bytes==alias.len()),
            "Q5 alias identity"
        );
        ensure!(
            observed.page_bytes == root.page_bytes
                && observed.returned_count == u64::from(PAGES_PER_ROOT)
                && observed.dispositions.len() == PAGES_PER_ROOT as usize
                && observed.raw_mach_status == 0
                && observed.elapsed_ns <= LIMIT_NS
                && observed.end_ns.checked_sub(observed.start_ns) == Some(observed.elapsed_ns),
            "Q5 alias partial/stale"
        );
        identity.matches(&file.metadata()?)?;
        let queried_bytes = root.queried_bytes;
        Ok(RootObservation::PartialNoCredit {
            root_id: id,
            requested_bytes,
            same_fd_device: identity.device,
            same_fd_inode: identity.inode,
            file_offset: registered_file_offset,
            queried_bytes,
            root: raw(root),
            ephemeral_alias: raw(observed),
        })
    };
    observe().unwrap_or(RootObservation::UnknownFullCharge {
        reason: "OwnedUncertifiedInvalidStaleOrChangedFD",
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn actual_result_gate_refuses_stale_partial_and_wrong_root() {
        let mut o = PageObservation {
            source: PageObservationSource::NoCopyRoot {
                root_id: 1,
                requested_bytes: 4096,
                registered_base: 4096,
                registered_bytes: 4096,
            },
            range: PageQueryRange {
                page_offset: 0,
                page_count: 1,
            },
            page_bytes: 4096,
            query_address: 4096,
            queried_bytes: 4096,
            raw_mach_status: 0,
            returned_count: 1,
            dispositions: vec![0],
            counts: mlx_native::PageDispositionCounts {
                present: 0,
                dirty: 0,
                wired: 0,
                copied: 0,
                paged_out: 0,
                other_known: 0,
            },
            clock: "test",
            start_ns: 1,
            end_ns: 2,
            elapsed_ns: 1,
        };
        validate(&o, 1, 4096, 1).unwrap();
        assert!(validate(&o, 2, 4096, 1).is_err());
        o.returned_count = 0;
        assert!(validate(&o, 1, 4096, 1).is_err());
        o.returned_count = 1;
        o.elapsed_ns = LIMIT_NS + 1;
        o.end_ns = o.start_ns + o.elapsed_ns;
        assert!(validate(&o, 1, 4096, 1).is_err());
        o.elapsed_ns = 1;
        o.end_ns = 2;
        o.dispositions[0] = 0x4000;
        assert!(validate(&o, 1, 4096, 1).is_err());
    }
}

// API availability is distinct from complete requested-domain authority.
pub(super) trait PublishedPageObserver {
    fn require_runtime_support(&self) -> Result<()>;
}
pub(super) struct Unavailable;
impl PublishedPageObserver for Unavailable {
    fn require_runtime_support(&self) -> Result<()> {
        anyhow::bail!("native_page_observer=Public14Available_WholeDomainAuthorityUnavailable")
    }
}
