//! Source weight-parent census only. File union is NOT physical deduplication.
use anyhow::{ensure, Context, Result};
use serde::Serialize;
#[derive(Serialize)]
pub(super) struct Coverage {
    scope: &'static str,
    unique_observed_weight_roots: usize,
    full_requested_bytes: usize,
    joined_file_parents: usize,
    unjoined_parents_full_charge: usize,
    mapped_logical_file_byte_union_bytes: u64,
    registered_file_page_union_bytes: u64,
    physical_credit_bytes: u64,
    whole_native_domain: &'static str,
}
fn union(mut spans: Vec<(u64, u64)>) -> Result<u64> {
    ensure!(spans.len() <= 16_384, "Q5 file-union capacity");
    spans.sort_unstable();
    let mut current: Option<(u64, u64)> = None;
    let mut sum = 0u64;
    for (start, len) in spans {
        let end = start
            .checked_add(len)
            .context("Q5 file interval overflow")?;
        ensure!(len > 0, "Q5 empty file interval");
        match current {
            Some((a, b)) if start <= b => current = Some((a, b.max(end))),
            Some((a, b)) => {
                sum = sum.checked_add(b - a).context("Q5 union overflow")?;
                current = Some((start, end));
            }
            None => current = Some((start, end)),
        }
    }
    if let Some((a, b)) = current {
        sum = sum.checked_add(b - a).context("Q5 union overflow")?;
    }
    Ok(sum)
}
/// Source metadata tuple is dev/ino/registered-start/registered-length/map-start/map-length.
/// A logical mmap is within EOF. Its OS covering pages can extend only to its
/// actual host-page-rounded end. Whole requested parent charge is unchanged.
pub(super) fn checked_spans(
    expected: crate::serve::q5_deepseek::packet::FileIdentity,
    page: u64,
    requested: usize,
    span: (u64, u64, u64, u64, u64, u64),
) -> Result<((u64, u64), (u64, u64))> {
    let (device, inode, start, len, mapping, mapping_len) = span;
    ensure!(
        device == expected.device && inode == expected.inode,
        "Q5 file interval source mismatch"
    );
    ensure!(
        (4096..=65536).contains(&page) && page.is_power_of_two(),
        "Q5 actual page geometry invalid"
    );
    ensure!(
        mapping_len > 0
            && len > 0
            && len == u64::try_from(requested)?
            && mapping % page == 0
            && start % page == 0
            && len % page == 0
            && start >= mapping,
        "Q5 requested/registered/mapping alignment or subset mismatch"
    );
    let mapping_end = mapping
        .checked_add(mapping_len)
        .context("Q5 logical mmap overflow")?;
    ensure!(mapping_end <= expected.bytes, "Q5 logical mmap outside FD");
    let rounded_end = mapping_end
        .checked_add(page - 1)
        .map(|n| n / page * page)
        .context("Q5 mapped page rounding overflow")?;
    let registered_end = start
        .checked_add(len)
        .context("Q5 registered range overflow")?;
    ensure!(
        registered_end <= rounded_end,
        "Q5 registered tail exceeds actual mapping covering page"
    );
    Ok(((mapping, mapping_len), (start, len)))
}
pub(super) fn actual(c: &super::Collector) -> Result<Coverage> {
    ensure!(
        c.weights.len() <= 16_384,
        "Q5 weight coverage root capacity"
    );
    let mut mapped_spans = Vec::new();
    mapped_spans.try_reserve_exact(c.weights.len())?;
    let mut registered_spans = Vec::new();
    registered_spans.try_reserve_exact(c.weights.len())?;
    let mut requested = 0usize;
    let mut joined = 0usize;
    for root in c.weights.values() {
        requested = requested
            .checked_add(root.bytes)
            .context("Q5 requested weight census overflow")?;
        if let Some(span) = root.file_span {
            let source = c.source.as_ref().context("Q5 joined span without source")?;
            let page = root
                .file_page_bytes
                .context("Q5 missing observed host page")?;
            let (mapped, registered) = checked_spans(source.expected, page, root.bytes, span)?;
            mapped_spans.push(mapped);
            registered_spans.push(registered);
            joined += 1;
        }
    }
    Ok(Coverage {
        scope: "ActualObservedWeightParentsOnly",
        unique_observed_weight_roots: c.weights.len(),
        full_requested_bytes: requested,
        joined_file_parents: joined,
        unjoined_parents_full_charge: c.weights.len() - joined,
        mapped_logical_file_byte_union_bytes: union(mapped_spans)?,
        registered_file_page_union_bytes: union(registered_spans)?,
        physical_credit_bytes: 0,
        whole_native_domain: "UnavailableUnlessSeparatelyCaptured_NoSubtractionFromRequested",
    })
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn actual_interval_union_does_not_sum_overlaps_and_rejects_overflow() {
        assert_eq!(union(vec![(8, 8), (0, 12), (0, 12), (20, 4)]).unwrap(), 20);
        assert!(union(vec![(u64::MAX, 1)]).is_err());
        assert!(union(vec![(0, 0)]).is_err());
        assert!(union(vec![(0, 1); 16_385]).is_err());
    }

    fn source(bytes: u64) -> crate::serve::q5_deepseek::packet::FileIdentity {
        crate::serve::q5_deepseek::packet::FileIdentity {
            device: 1,
            inode: 2,
            bytes,
            mtime_seconds: 0,
            mtime_nanoseconds: 0,
            ctime_seconds: 0,
            ctime_nanoseconds: 0,
        }
    }
    #[test]
    fn actual_extent_gate_accepts_only_justified_page_tail_and_subsets() {
        let (mapped, registered) =
            checked_spans(source(101), 4096, 4096, (1, 2, 0, 4096, 0, 101)).unwrap();
        assert_eq!(mapped, (0, 101));
        assert_eq!(registered, (0, 4096));
        assert_eq!(union(vec![mapped, mapped]).unwrap(), 101);
        assert_eq!(union(vec![registered, registered]).unwrap(), 4096);
        assert_eq!(registered.1, 4096); // full requested parent remains charged, not clipped101.
        checked_spans(source(8192), 4096, 4096, (1, 2, 4096, 4096, 0, 8192)).unwrap();
    }
    #[test]
    fn actual_extent_gate_refuses_bad_eof_tail_geometry_and_identity() {
        for span in [
            (1, 2, 0, 4096, 0, 102),
            (1, 2, 0, 8192, 0, 101),
            (1, 2, 0, 4096, 4096, 1),
            (1, 2, 1, 4096, 0, 101),
            (3, 2, 0, 4096, 0, 101),
            (1, 3, 0, 4096, 0, 101),
        ] {
            let requested = span.3 as usize;
            assert!(checked_spans(source(101), 4096, requested, span).is_err());
        }
        assert!(checked_spans(source(101), 4096, 8192, (1, 2, 0, 4096, 0, 101)).is_err());
        assert!(checked_spans(source(101), 3000, 4096, (1, 2, 0, 4096, 0, 101)).is_err());
        assert!(checked_spans(
            source(u64::MAX),
            4096,
            4096,
            (1, 2, u64::MAX - 4095, 4096, u64::MAX - 4095, 4096)
        )
        .is_err());
        assert!(checked_spans(source(u64::MAX), 4096, 4096, (1, 2, 0, 4096, 0, u64::MAX)).is_err());
    }
}
