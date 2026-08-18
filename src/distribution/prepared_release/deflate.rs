//! Exact raw-Deflate framing checks for the deliberately narrow ZIP profile.

use std::io::{Read, Seek, SeekFrom};

use flate2::{Decompress, FlushDecompress, Status};

use super::archive::ClassicEntry;
use super::PreparedReleaseError;

/// Require every declared compressed range to contain exactly one complete
/// raw-Deflate stream. High-level `Read` adapters are insufficient here: at
/// source EOF they may report ordinary EOF without proving `StreamEnd`.
pub(super) fn verify_exact_streams<R: Read + Seek>(
    reader: &mut R,
    entries: &[ClassicEntry],
) -> Result<(), PreparedReleaseError> {
    let mut input = [0_u8; 64 * 1024];
    let mut output = [0_u8; 64 * 1024];
    for entry in entries.iter().filter(|entry| entry.method == 8) {
        reader
            .seek(SeekFrom::Start(entry.data_start))
            .map_err(|_| PreparedReleaseError::ArchiveRead)?;
        let mut decoder = Decompress::new(false);
        let mut remaining = u64::from(entry.compressed_size);
        let mut stream_end = false;
        while remaining != 0 {
            let count = usize::try_from(remaining.min(input.len() as u64))
                .map_err(|_| PreparedReleaseError::ArchiveProfile)?;
            reader
                .read_exact(&mut input[..count])
                .map_err(|_| PreparedReleaseError::ArchiveRead)?;
            remaining -= count as u64;
            let mut offset = 0_usize;
            while offset < count {
                let before_in = decoder.total_in();
                let before_out = decoder.total_out();
                let status = decoder
                    .decompress(&input[offset..count], &mut output, FlushDecompress::None)
                    .map_err(|_| PreparedReleaseError::ArchiveProfile)?;
                let consumed = usize::try_from(decoder.total_in() - before_in)
                    .map_err(|_| PreparedReleaseError::ArchiveProfile)?;
                let produced = decoder.total_out() - before_out;
                offset = offset
                    .checked_add(consumed)
                    .ok_or(PreparedReleaseError::ArchiveProfile)?;
                if decoder.total_out() > u64::from(entry.uncompressed_size)
                    || (consumed == 0 && produced == 0)
                {
                    return Err(PreparedReleaseError::ArchiveProfile);
                }
                if status == Status::StreamEnd {
                    if offset != count || remaining != 0 {
                        return Err(PreparedReleaseError::ArchiveProfile);
                    }
                    stream_end = true;
                    break;
                }
            }
            if stream_end {
                break;
            }
        }
        while !stream_end {
            let before_out = decoder.total_out();
            let status = decoder
                .decompress(&[], &mut output, FlushDecompress::Finish)
                .map_err(|_| PreparedReleaseError::ArchiveProfile)?;
            if decoder.total_out() > u64::from(entry.uncompressed_size) {
                return Err(PreparedReleaseError::ArchiveProfile);
            }
            if status == Status::StreamEnd {
                stream_end = true;
            } else if decoder.total_out() == before_out {
                return Err(PreparedReleaseError::ArchiveProfile);
            }
        }
        if decoder.total_in() != u64::from(entry.compressed_size)
            || decoder.total_out() != u64::from(entry.uncompressed_size)
        {
            return Err(PreparedReleaseError::ArchiveProfile);
        }
    }
    Ok(())
}
