//! Actual loaded tensor/retained FD association and bounded public page metadata.
use super::{with_current, Domain};
use crate::serve::q5_deepseek::page_observer;
use anyhow::{ensure, Context, Result};
use mlx_native::{MlxBuffer, RootOriginSnapshot};
pub(crate) fn weight(
    gguf: &mlx_native::gguf::GgufFile,
    name: &str,
    buffer: &MlxBuffer,
) -> Result<()> {
    with_current(|c| {
        let source = c
            .source
            .as_ref()
            .context("Q5 weight missing actual SameFD owner")?;
        source.expected.matches(&source.file.metadata()?)?;
        ensure!(
            source.tensor_data_offset == gguf.tensor_data_offset(),
            "Q5 loader tensor offset drift"
        );
        let info = gguf.tensor_info(name).context("Q5 actual tensor missing")?;
        let logical = source
            .tensor_data_offset
            .checked_add(info.offset)
            .context("Q5 file offset overflow")?;
        ensure!(
            logical
                .checked_add(info.byte_len as u64)
                .is_some_and(|end| end <= source.expected.bytes),
            "Q5 tensor escapes SameFD extent"
        );
        let b = buffer.backing_snapshot()?;
        let file_span = match b.provenance {
            mlx_native::BufferProvenance::NoCopy => {
                let n = b
                    .no_copy
                    .context("Q5 NoCopy missing actual mapping range")?;
                let within = n
                    .registered_base_address
                    .checked_sub(n.mapping_base_address)
                    .context("Q5 NoCopy range order")?;
                let registered = logical
                    .checked_sub(b.view_byte_offset)
                    .context("Q5 registered offset underflow")?;
                let mapping = registered
                    .checked_sub(within as u64)
                    .context("Q5 mapping offset underflow")?;
                ensure!(
                    b.logical_bytes == info.byte_len
                        && n.registered_bytes == b.metal_root_bytes
                        && mapping
                            .checked_add(n.mapping_bytes as u64)
                            .is_some_and(|end| end <= source.expected.bytes),
                    "Q5 actual mapped span mismatch"
                );
                Some((
                    source.expected.device,
                    source.expected.inode,
                    registered,
                    n.registered_bytes as u64,
                    mapping,
                    n.mapping_bytes as u64,
                ))
            }
            mlx_native::BufferProvenance::Owned => None,
            _ => anyhow::bail!("Q5 Unknown weight provenance"),
        };
        // SAFETY: fixed sysconf metadata query, no pointers or payload access.
        let file_page_bytes = if file_span.is_some() {
            let page = unsafe { libc::sysconf(libc::_SC_PAGESIZE) };
            let page = u64::try_from(page).context("Q5 actual host page unavailable")?;
            ensure!(
                (4096..=65536).contains(&page) && page.is_power_of_two(),
                "Q5 unsupported actual host page"
            );
            let n = b.no_copy.context("Q5 NoCopy descriptor disappeared")?;
            ensure!(
                n.mapping_base_address as u64 % page == 0
                    && n.registered_base_address as u64 % page == 0,
                "Q5 actual NoCopy CPU base page alignment"
            );
            Some(page)
        } else {
            None
        };
        source.expected.matches(&source.file.metadata()?)?;
        let observation_file = source.file.try_clone()?;
        let observation_identity = source.expected;
        c.buffer(Some(Domain::Weight), name, buffer)?;
        if let RootOriginSnapshot::KnownNative { id, .. } = b.origin {
            let root = c
                .weights
                .get_mut(&id)
                .context("Q5 actual weight root census missing")?;
            ensure!(
                root.file_span.is_none() || root.file_span == file_span,
                "Q5 one root changed actual SameFD span"
            );
            ensure!(
                root.file_page_bytes.is_none() || root.file_page_bytes == file_page_bytes,
                "Q5 one root changed actual host page"
            );
            if let Some(span) = file_span {
                super::coverage::checked_spans(
                    observation_identity,
                    file_page_bytes.context("Q5 missing actual page")?,
                    root.bytes,
                    span,
                )?;
            }
            root.file_span = file_span;
            root.file_page_bytes = file_page_bytes;
            if !c.page_observations.contains_key(&id)
                && c.page_observations.len() < page_observer::MAX_ROOT_QUERIES
            {
                let observation = match file_span {
                    Some((_, _, offset, extent, _, _)) => {
                        let overlap = c.page_observations.iter().any(|(other, _)| {
                            *other != id
                                && c.weights.get(other).and_then(|r| r.file_span).is_some_and(
                                    |(_, _, start, len, _, _)| {
                                        offset
                                            .checked_add(extent)
                                            .zip(start.checked_add(len))
                                            .is_none_or(|(end, other_end)| {
                                                offset < other_end && start < end
                                            })
                                    },
                                )
                        });
                        if overlap {
                            page_observer::RootObservation::UnknownFullCharge {
                                reason: "OverlappingRegisteredFileSpans",
                            }
                        } else {
                            page_observer::actual(
                                buffer,
                                &observation_file,
                                observation_identity,
                                offset,
                                id,
                                b.metal_root_bytes,
                            )
                        }
                    }
                    None => page_observer::RootObservation::UnknownFullCharge {
                        reason: "OwnedNoFilePageCoverage",
                    },
                };
                c.page_observations.insert(id, observation);
            }
        }
        Ok(())
    })
}
