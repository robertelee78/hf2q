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
    registered_file_interval_union_bytes: u64,
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
pub(super) fn actual(c: &super::Collector) -> Result<Coverage> {
    let mut spans = Vec::new();
    spans.try_reserve_exact(c.weights.len())?;
    let mut requested = 0usize;
    let mut joined = 0usize;
    for root in c.weights.values() {
        requested = requested
            .checked_add(root.bytes)
            .context("Q5 requested weight census overflow")?;
        if let Some((device, inode, start, len, _, _)) = root.file_span {
            let source = c.source.as_ref().context("Q5 joined span without source")?;
            ensure!(
                device == source.expected.device && inode == source.expected.inode,
                "Q5 file interval source mismatch"
            );
            // Rounded registered tail may exceed EOF; it is not valid file-union coverage.
            ensure!(
                start
                    .checked_add(len)
                    .is_some_and(|end| end <= source.expected.bytes),
                "Q5 registered file interval outside FD"
            );
            spans.push((start, len));
            joined += 1;
        }
    }
    Ok(Coverage {
        scope: "ActualObservedWeightParentsOnly",
        unique_observed_weight_roots: c.weights.len(),
        full_requested_bytes: requested,
        joined_file_parents: joined,
        unjoined_parents_full_charge: c.weights.len() - joined,
        registered_file_interval_union_bytes: union(spans)?,
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
}
