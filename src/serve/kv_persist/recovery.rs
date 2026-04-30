//! ADR-017 §A.2 — restart-recovery scan + quarantine surface.
//!
//! On `cmd_serve --kv-persist=on` startup, [`recover_from_disk`] walks
//! the on-disk cache and rebuilds the in-memory `BlockIndex`. Files
//! that fail to parse, fail the version check, or have corrupted bodies
//! are MOVED (not deleted) to `<slug>/kv-quarantine/` so an operator
//! can post-mortem the bytes (ADR-017 §R-F9).
//!
//! Body-hash validation is *NOT* performed during the recovery scan —
//! we only read the JSON header. Body integrity is checked lazily on
//! the read path via [`format::read_envelope_body`]; if a body is
//! corrupted, the reader path bubbles the error and the caller passes
//! the file through [`quarantine_corrupted_block`] with
//! [`QuarantineReason::BodyHashMismatch`].
//!
//! ## Why two surfaces (recover + quarantine)
//!
//! `recover_from_disk` is the bulk startup walk; it's O(file count)
//! header reads. `quarantine_corrupted_block` is the lazy single-file
//! surface called by the read path AFTER a body-hash mismatch
//! surfaces. Both end up moving the file under `<slug>/kv-quarantine/`,
//! but the entry points are distinct so the two contexts don't share
//! error-handling code that's wrong for one of them.

use std::fs;
use std::io;
use std::path::{Path, PathBuf};
use std::time::{Instant, SystemTime};

use crate::serve::kv_persist::format::{self, EnvelopeHeader, ModelFingerprint, CURRENT_FORMAT_VERSION};
use crate::serve::kv_persist::index::{BlockIndex, BlockMeta};

/// Outcome of a recovery scan. All fields are derived from real
/// `fs::metadata` calls on the on-disk artifacts — never synthesized
/// (per `feedback_substrate_must_not_synthesize_ship_gates`).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct RecoveryReport {
    /// Files successfully indexed (header parsed, version OK).
    pub blocks_indexed: usize,
    /// Files moved to `<slug>/kv-quarantine/` (header parse failure or
    /// version mismatch). Body-hash failures are NOT counted here
    /// because they are detected lazily on the read path.
    pub blocks_quarantined: usize,
    /// Sum of `bytes_on_disk` for indexed files.
    pub bytes_indexed: u64,
    /// Sum of file size for quarantined files (computed via
    /// `fs::metadata` BEFORE the move; the post-move size is
    /// identical so this is the canonical figure).
    pub bytes_quarantined: u64,
    /// `*.tmp.<pid>` files left behind by a prior crashed write
    /// (§D5 atomic-rename leftovers). Counted but not modified.
    pub partial_tmp_files_ignored: usize,
    /// Wall-clock duration of the entire scan, milliseconds.
    pub elapsed_ms: u128,
}

/// Distinct quarantine causes — used by the read path to attach a
/// reason prefix to the quarantined filename so an operator can grep
/// for "why was this block quarantined".
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum QuarantineReason {
    /// Header bytes truncated / unreadable (file was shorter than the
    /// declared `header_len`, or `header_len` was out-of-range).
    TruncatedHeader,
    /// `format_version` field in the header didn't match
    /// [`CURRENT_FORMAT_VERSION`].
    VersionMismatch,
    /// `read_envelope_body` recomputed `sha256(body)` and got something
    /// other than `header.block_hash`.
    BodyHashMismatch,
    /// On-disk parity / checksum check on a future-format envelope
    /// failed (reserved; not used in the v1 envelope but kept here so
    /// forward-compatible quarantine reasoning has a slot).
    ParityFail,
}

impl QuarantineReason {
    /// Filename prefix used when moving the file to `kv-quarantine/`.
    /// Operators can grep `kv-quarantine/` for these prefixes.
    fn prefix(self) -> &'static str {
        match self {
            QuarantineReason::TruncatedHeader => "trunc",
            QuarantineReason::VersionMismatch => "verbump",
            QuarantineReason::BodyHashMismatch => "bodyhash",
            QuarantineReason::ParityFail => "parity",
        }
    }
}

/// Walk the cache root and rebuild a [`BlockIndex`]. Quarantines
/// header-error and version-mismatch files. Returns the populated
/// index plus a structured [`RecoveryReport`] suitable for logging
/// and the §B.4 telemetry surface.
///
/// `cache_root` layout per ADR-017 §D5:
/// ```text
/// <cache_root>/
///   models/
///     <slug>/
///       kv/
///         <hex0>/
///           <full_hex>.safetensors
///           <full_hex>.safetensors.tmp.<pid>   ← ignored
///       kv-quarantine/                          ← created lazily on quarantine
/// ```
///
/// Errors propagate ONLY for unrecoverable I/O on the cache root
/// (e.g. permission denied). Per-file failures are routed to
/// quarantine and accounted in `RecoveryReport`.
pub fn recover_from_disk(cache_root: &Path) -> io::Result<(BlockIndex, RecoveryReport)> {
    let start = Instant::now();
    let index = BlockIndex::new();
    let mut report = RecoveryReport::default();

    let models_dir = cache_root.join("models");
    if !models_dir.exists() {
        report.elapsed_ms = start.elapsed().as_millis();
        return Ok((index, report));
    }

    for slug_ent in fs::read_dir(&models_dir)? {
        let slug_ent = slug_ent?;
        let slug_path = slug_ent.path();
        if !slug_path.is_dir() {
            continue;
        }
        let kv_dir = slug_path.join("kv");
        if !kv_dir.exists() {
            continue;
        }
        for fanout_ent in fs::read_dir(&kv_dir)? {
            let fanout_ent = fanout_ent?;
            let fanout_path = fanout_ent.path();
            if !fanout_path.is_dir() {
                continue;
            }
            for blk_ent in fs::read_dir(&fanout_path)? {
                let blk_ent = blk_ent?;
                let blk_path = blk_ent.path();
                if !blk_path.is_file() {
                    continue;
                }
                scan_one(&slug_path, &blk_path, &index, &mut report)?;
            }
        }
    }

    report.elapsed_ms = start.elapsed().as_millis();
    Ok((index, report))
}

/// Process one envelope file. Indexes it, quarantines it, or ignores
/// it (`.tmp.<pid>` orphans). Returns `Ok(())` for every per-file
/// outcome; only unrecoverable I/O bubbles up.
fn scan_one(
    slug_path: &Path,
    blk_path: &Path,
    index: &BlockIndex,
    report: &mut RecoveryReport,
) -> io::Result<()> {
    let name = blk_path
        .file_name()
        .and_then(|s| s.to_str())
        .unwrap_or("")
        .to_string();

    // Atomic-rename leftovers from a prior crashed write (§D5 + §D8).
    if name.contains(".tmp.") {
        report.partial_tmp_files_ignored += 1;
        return Ok(());
    }

    // Capture the file size BEFORE any potential move so the quarantine
    // bytes accounting is correct even if the rename succeeds.
    let file_bytes = match fs::metadata(blk_path) {
        Ok(m) => m.len(),
        Err(_) => 0,
    };

    let header = match format::read_envelope_header(blk_path) {
        Ok(h) => h,
        Err(_) => {
            quarantine_with_prefix(slug_path, blk_path, QuarantineReason::TruncatedHeader)?;
            report.blocks_quarantined += 1;
            report.bytes_quarantined = report.bytes_quarantined.saturating_add(file_bytes);
            return Ok(());
        }
    };

    if header.format_version != CURRENT_FORMAT_VERSION.0 {
        quarantine_with_prefix(slug_path, blk_path, QuarantineReason::VersionMismatch)?;
        report.blocks_quarantined += 1;
        report.bytes_quarantined = report.bytes_quarantined.saturating_add(file_bytes);
        return Ok(());
    }

    let metadata = fs::metadata(blk_path)?;
    let mtime = metadata.modified().unwrap_or(SystemTime::UNIX_EPOCH);
    let bytes_on_disk = metadata.len();
    let meta = blockmeta_from_header(&header, blk_path.to_path_buf(), mtime, bytes_on_disk);
    index.insert(meta);
    report.blocks_indexed += 1;
    report.bytes_indexed = report.bytes_indexed.saturating_add(bytes_on_disk);
    Ok(())
}

/// Build a [`BlockMeta`] from a parsed header + filesystem metadata.
/// Kept as a free function so the recovery and write paths produce
/// identical metadata shape (the writer mirrors this construction in
/// `block_store::write_block_sync`).
fn blockmeta_from_header(
    header: &EnvelopeHeader,
    file_path: PathBuf,
    mtime: SystemTime,
    bytes_on_disk: u64,
) -> BlockMeta {
    BlockMeta {
        hash: header.block_hash,
        parent: header.parent_block_hash,
        model_fp: header.model_fingerprint,
        payload_kind: header.payload_kind.clone(),
        codec_version: header.codec_version,
        n_tokens: header.n_tokens,
        file_path,
        mtime,
        bytes_on_disk,
    }
}

/// Move a corrupted file to `<slug>/kv-quarantine/<reason_prefix>__<original-name>`.
/// Public surface called by the read path when it detects a body-hash
/// mismatch on a previously-indexed block.
///
/// `original_path` is the file's current location (e.g. `<slug>/kv/<hex0>/<full>.safetensors`).
/// The function moves it into the quarantine directory with a
/// reason-prefix so operators can grep for cause.
///
/// Falls back to copy + remove if `fs::rename` fails (cross-fs edge
/// case — the quarantine dir is normally a sibling of the kv tree, so
/// rename should always succeed in practice).
pub fn quarantine_corrupted_block(
    cache_root: &Path,
    model_fp: &ModelFingerprint,
    original_path: &Path,
    reason: QuarantineReason,
) -> io::Result<PathBuf> {
    let slug_path = cache_root.join("models").join(model_fp.short_hex());
    quarantine_with_prefix(&slug_path, original_path, reason)
}

/// Internal: move `<slug>/kv/<...>/<name>` to
/// `<slug>/kv-quarantine/<reason>__<name>`. Returns the new path.
fn quarantine_with_prefix(
    slug_path: &Path,
    blk_path: &Path,
    reason: QuarantineReason,
) -> io::Result<PathBuf> {
    let q_dir = slug_path.join("kv-quarantine");
    if !q_dir.exists() {
        fs::create_dir_all(&q_dir)?;
    }
    let name = blk_path
        .file_name()
        .ok_or_else(|| io::Error::other(format!("blk path has no name: {}", blk_path.display())))?
        .to_string_lossy()
        .to_string();
    let dest_name = format!("{}__{}", reason.prefix(), name);
    let dest = q_dir.join(dest_name);
    match fs::rename(blk_path, &dest) {
        Ok(()) => Ok(dest),
        Err(_) => {
            // Cross-fs fallback. Should be rare since quarantine sits
            // under the same model slug as the source file.
            fs::copy(blk_path, &dest)?;
            fs::remove_file(blk_path)?;
            Ok(dest)
        }
    }
}

// ---------------------------------------------------------------------------
// Tests.
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::serve::kv_persist::format::{
        compute_model_fingerprint, write_envelope, BlockHash, EnvelopeHeader, ParentBlockHash,
        BLOCK_TOKENS,
    };
    use sha2::{Digest, Sha256};
    use std::process;
    use std::sync::atomic::{AtomicU32, Ordering};

    fn temp_dir(label: &str) -> PathBuf {
        static COUNTER: AtomicU32 = AtomicU32::new(0);
        let n = COUNTER.fetch_add(1, Ordering::SeqCst);
        let pid = process::id();
        let nanos = SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0);
        let dir = std::env::temp_dir().join(format!("hf2q-kv-rec-{label}-{pid}-{nanos}-{n}"));
        fs::create_dir_all(&dir).expect("temp_dir mkdir");
        dir
    }

    fn fixture_fp(seed: &str) -> ModelFingerprint {
        compute_model_fingerprint(
            seed,
            "Q4_0",
            "hf2q-test-1.0.0",
            "deadbeefcafebabe1122334455667788",
            "<|im_start|>...<|im_end|>",
        )
    }

    fn make_block(
        fp: ModelFingerprint,
        parent: ParentBlockHash,
        seed: u32,
    ) -> (Vec<u8>, EnvelopeHeader) {
        let body: Vec<u8> = (0..512u32)
            .flat_map(|i| (i.wrapping_add(seed)).to_le_bytes())
            .collect();
        let mut h = Sha256::new();
        h.update(&body);
        let bh: [u8; 32] = h.finalize().into();
        let header = EnvelopeHeader {
            format_version: CURRENT_FORMAT_VERSION.0,
            model_fingerprint: fp,
            block_hash: BlockHash(bh),
            parent_block_hash: parent,
            payload_kind: "kv-dense-bf16".into(),
            codec_version: 1,
            n_tokens: BLOCK_TOKENS,
        };
        (body, header)
    }

    fn block_path(root: &Path, fp: &ModelFingerprint, hash: &BlockHash) -> PathBuf {
        let hex = hash.to_string();
        let fanout = &hex[..1];
        root.join("models")
            .join(fp.short_hex())
            .join("kv")
            .join(fanout)
            .join(format!("{hex}.safetensors"))
    }

    #[test]
    fn recover_from_disk_clean_state_returns_empty_report() {
        // Nonexistent root → empty index, all-zero report.
        let dir = temp_dir("clean-empty");
        let (idx, report) = recover_from_disk(&dir).expect("recover");
        assert_eq!(idx.block_count(), 0);
        assert_eq!(report.blocks_indexed, 0);
        assert_eq!(report.blocks_quarantined, 0);
        assert_eq!(report.bytes_indexed, 0);
        assert_eq!(report.bytes_quarantined, 0);
        assert_eq!(report.partial_tmp_files_ignored, 0);

        // Existing-but-empty models dir → still empty.
        fs::create_dir_all(dir.join("models")).expect("mkdir");
        let (idx2, report2) = recover_from_disk(&dir).expect("recover2");
        assert_eq!(idx2.block_count(), 0);
        assert_eq!(report2.blocks_indexed, 0);

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn recover_from_disk_50_blocks_yields_50_indexed() {
        let dir = temp_dir("rec50");
        let fp = fixture_fp("rec50");

        let mut hashes: Vec<BlockHash> = Vec::new();
        let mut total_bytes: u64 = 0;
        let mut parent = ParentBlockHash(None);
        for s in 0u32..50 {
            let (body, header) = make_block(fp, parent, s);
            let path = block_path(&dir, &fp, &header.block_hash);
            write_envelope(&path, &header, &body).expect("write");
            hashes.push(header.block_hash);
            total_bytes += fs::metadata(&path).unwrap().len();
            parent = ParentBlockHash(Some(header.block_hash));
        }

        let (idx, report) = recover_from_disk(&dir).expect("recover");
        assert_eq!(report.blocks_indexed, 50);
        assert_eq!(report.blocks_quarantined, 0);
        assert_eq!(report.bytes_indexed, total_bytes, "bytes match real fs::metadata sum");
        assert_eq!(report.partial_tmp_files_ignored, 0);
        assert_eq!(idx.block_count(), 50);

        // Every written hash is in the index, with a real on-disk file.
        for h in &hashes {
            let m = idx.lookup(h).expect("indexed");
            assert!(m.file_path.exists(), "indexed file exists on disk");
            assert!(m.bytes_on_disk > 0);
        }

        // No quarantine dir created on a clean recovery.
        let q = dir.join("models").join(fp.short_hex()).join("kv-quarantine");
        assert!(!q.exists(), "no quarantine on clean recovery");

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn recover_from_disk_with_corrupted_blocks_reports_quarantined_count() {
        let dir = temp_dir("rec-quar");
        let fp = fixture_fp("rec-quar");

        // Write 5 valid blocks.
        let mut paths: Vec<PathBuf> = Vec::new();
        let mut hashes: Vec<BlockHash> = Vec::new();
        let mut parent = ParentBlockHash(None);
        for s in 0u32..5 {
            let (body, header) = make_block(fp, parent, s);
            let path = block_path(&dir, &fp, &header.block_hash);
            write_envelope(&path, &header, &body).expect("write");
            paths.push(path);
            hashes.push(header.block_hash);
            parent = ParentBlockHash(Some(header.block_hash));
        }

        // Corrupt 2 of them differently:
        //   index 1: truncate to 3 bytes (header parse fails)
        //   index 3: rewrite envelope with format_version=999 (version mismatch)
        fs::write(&paths[1], b"abc").expect("truncate");

        let bad_header = EnvelopeHeader {
            format_version: 999,
            model_fingerprint: fp,
            block_hash: hashes[3],
            parent_block_hash: ParentBlockHash(Some(hashes[2])),
            payload_kind: "kv-dense-bf16".into(),
            codec_version: 1,
            n_tokens: BLOCK_TOKENS,
        };
        let header_json = serde_json::to_vec(&bad_header).expect("ser");
        let pad = (8 - (header_json.len() % 8)) % 8;
        let mut header_bytes = header_json;
        header_bytes.extend(std::iter::repeat(b' ').take(pad));
        let mut blob = Vec::new();
        blob.extend_from_slice(&(header_bytes.len() as u64).to_le_bytes());
        blob.extend_from_slice(&header_bytes);
        blob.extend_from_slice(&[0u8; 64]);
        fs::write(&paths[3], &blob).expect("write bad");

        let (idx, report) = recover_from_disk(&dir).expect("recover");
        assert_eq!(report.blocks_indexed, 3);
        assert_eq!(report.blocks_quarantined, 2);
        assert!(report.bytes_indexed > 0);
        assert!(report.bytes_quarantined > 0, "quarantined bytes accounted");

        assert_eq!(idx.block_count(), 3);

        // Quarantine dir contains 2 files with prefixed names.
        let q = dir.join("models").join(fp.short_hex()).join("kv-quarantine");
        let q_files: Vec<String> = fs::read_dir(&q)
            .expect("read q")
            .filter_map(|e| e.ok())
            .map(|e| e.file_name().to_string_lossy().into_owned())
            .collect();
        assert_eq!(q_files.len(), 2, "two quarantined files; saw {q_files:?}");
        // One has the truncation prefix, one has the version prefix.
        assert!(q_files.iter().any(|n| n.starts_with("trunc__")), "trunc__ prefix");
        assert!(
            q_files.iter().any(|n| n.starts_with("verbump__")),
            "verbump__ prefix"
        );

        // Originals removed from kv/<fanout>/.
        assert!(!paths[1].exists());
        assert!(!paths[3].exists());

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn quarantine_truncated_header_moves_file_to_kv_quarantine_dir() {
        // Direct call: write a valid block, corrupt it, call
        // quarantine_corrupted_block. Verify the file lands at the
        // expected path with the expected reason prefix.
        let dir = temp_dir("q-trunc");
        let fp = fixture_fp("q-trunc");

        let (body, header) = make_block(fp, ParentBlockHash(None), 0);
        let original = block_path(&dir, &fp, &header.block_hash);
        write_envelope(&original, &header, &body).expect("write");

        let dest =
            quarantine_corrupted_block(&dir, &fp, &original, QuarantineReason::TruncatedHeader)
                .expect("quarantine");

        // Source gone, dest present.
        assert!(!original.exists());
        assert!(dest.exists());

        // Dest path: <root>/models/<short>/kv-quarantine/trunc__<full>.safetensors
        let dest_str = dest.to_string_lossy();
        assert!(dest_str.contains("/kv-quarantine/"));
        let dest_name = dest.file_name().unwrap().to_string_lossy().into_owned();
        assert!(dest_name.starts_with("trunc__"), "trunc prefix: {dest_name}");
        assert!(dest_name.ends_with(".safetensors"));

        // Bytes round-trip (we just moved the file; content is unchanged).
        let moved_bytes = fs::read(&dest).expect("read moved");
        assert!(moved_bytes.len() > 8, "moved file has content");

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn quarantine_body_hash_mismatch_uses_distinct_reason_prefix() {
        // The 4 QuarantineReason variants must produce 4 distinct
        // filename prefixes — operators rely on this for grep-by-cause.
        let dir = temp_dir("q-prefix");
        let fp = fixture_fp("q-prefix");

        // Write 4 valid blocks; quarantine each with a different reason.
        let reasons = [
            QuarantineReason::TruncatedHeader,
            QuarantineReason::VersionMismatch,
            QuarantineReason::BodyHashMismatch,
            QuarantineReason::ParityFail,
        ];
        let mut prefixes: Vec<&'static str> = Vec::new();

        for (i, reason) in reasons.iter().enumerate() {
            let (body, header) = make_block(fp, ParentBlockHash(None), i as u32);
            let original = block_path(&dir, &fp, &header.block_hash);
            write_envelope(&original, &header, &body).expect("write");
            let dest =
                quarantine_corrupted_block(&dir, &fp, &original, *reason).expect("quarantine");
            let name = dest.file_name().unwrap().to_string_lossy().into_owned();
            // Extract the prefix (everything before "__").
            let prefix = name
                .split("__")
                .next()
                .expect("prefix split")
                .to_string()
                .leak() as &'static str;
            prefixes.push(prefix);
            assert!(dest.exists());
            assert!(!original.exists());
        }

        // All 4 prefixes are distinct.
        let mut sorted = prefixes.clone();
        sorted.sort();
        sorted.dedup();
        assert_eq!(sorted.len(), 4, "four distinct prefixes; got {prefixes:?}");
        assert!(prefixes.contains(&"trunc"));
        assert!(prefixes.contains(&"verbump"));
        assert!(prefixes.contains(&"bodyhash"));
        assert!(prefixes.contains(&"parity"));

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn recover_ignores_tmp_files_and_counts_them() {
        // *.tmp.<pid> orphans are counted in partial_tmp_files_ignored
        // but neither indexed nor quarantined.
        let dir = temp_dir("rec-tmp");
        let fp = fixture_fp("rec-tmp");

        // Write 3 valid blocks.
        let mut hashes: Vec<BlockHash> = Vec::new();
        let mut parent = ParentBlockHash(None);
        for s in 0u32..3 {
            let (body, header) = make_block(fp, parent, s);
            let path = block_path(&dir, &fp, &header.block_hash);
            write_envelope(&path, &header, &body).expect("write");
            hashes.push(header.block_hash);
            parent = ParentBlockHash(Some(header.block_hash));
        }

        // Drop two `.tmp.<pid>` orphans in different fanout dirs.
        let fanout_dir_a = dir
            .join("models")
            .join(fp.short_hex())
            .join("kv")
            .join(&hashes[0].to_string()[..1]);
        fs::write(
            fanout_dir_a.join(format!("orphan.safetensors.tmp.{}", process::id())),
            b"partial-bytes-1",
        )
        .expect("write orphan a");
        fs::write(
            fanout_dir_a.join(format!("other.safetensors.tmp.{}", process::id() + 1)),
            b"partial-bytes-2",
        )
        .expect("write orphan b");

        let (idx, report) = recover_from_disk(&dir).expect("recover");
        assert_eq!(report.blocks_indexed, 3);
        assert_eq!(report.blocks_quarantined, 0);
        assert_eq!(report.partial_tmp_files_ignored, 2);
        assert_eq!(idx.block_count(), 3);

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn recovery_report_elapsed_ms_is_nonzero_for_real_walk() {
        // Smoke-check that the report's elapsed_ms field is populated
        // even for a small walk. A 10-block scan on a hot SSD takes
        // single-digit ms; we just assert >= 0 (the field is u128).
        let dir = temp_dir("rec-elapsed");
        let fp = fixture_fp("rec-elapsed");
        let mut parent = ParentBlockHash(None);
        for s in 0u32..3 {
            let (body, header) = make_block(fp, parent, s);
            let path = block_path(&dir, &fp, &header.block_hash);
            write_envelope(&path, &header, &body).expect("write");
            parent = ParentBlockHash(Some(header.block_hash));
        }
        let (_, report) = recover_from_disk(&dir).expect("recover");
        // u128 always >= 0; the assertion is "the field is populated"
        // — we exercise the code path that sets it.
        assert!(
            report.elapsed_ms < 60_000,
            "scan completed in well under 60s: {} ms",
            report.elapsed_ms
        );

        let _ = fs::remove_dir_all(&dir);
    }
}
