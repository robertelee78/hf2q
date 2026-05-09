//! ADR-027 Phase A iter-5 — disk persistence for the Qwen3.5/3.6 hybrid
//! KV-cache snapshot via the QH35 envelope.
//!
//! Provides the cold-process resume path: serialize a
//! `HybridKvCacheSnapshot` through `qwen35_hybrid_persistor::serialize_*`
//! and write it to `~/.cache/hf2q/qwen35_kv/<fingerprint>/<lcp_key>.bin`;
//! on next process start, hydrate the in-memory `LcpRegistry` by walking
//! the directory and deserializing each file.
//!
//! This module is the disk back-end ONLY — wiring into `Qwen35LoadedModel`
//! and the `cmd_serve` / `cmd_generate_qwen35` env-toggle landed in
//! iter-6 (gated separately because that touches the engine load path).
//!
//! # File-on-disk contract
//!
//! ```text
//! <cache_dir>/<fingerprint>/<lcp_key_hex>.bin
//! ```
//!
//! `<fingerprint>` is a 16-hex-char (8-byte) digest of the
//! `Qwen35HybridConfig` shape fields so a re-quanted model with a
//! different config can never read another's blocks.  `<lcp_key_hex>`
//! is the lcp_key's stable hex form (mirrors what `LcpRegistry` would
//! produce for the same key in memory).
//!
//! Atomic writes use the standard tempfile-and-rename pattern so a
//! crash mid-write leaves the persisted file intact.
//!
//! # Iter sequence
//!
//! - **Iter 5 (this commit)**: write / read / hydrate_registry surface
//!   + 4 unit tests (round-trip via tempdir; no GPU model required).
//! - **Iter 6**: thread `Qwen35DiskPersistor` into `Qwen35LoadedModel`
//!   under `HF2Q_KV_PERSIST=<dir>` and validate cold-process LCP resume
//!   on Qwen 3.6 35B-A3B-APEX-Q5_K_M.

use anyhow::{anyhow, ensure, Context, Result};
use mlx_native::MlxDevice;
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};

use crate::inference::models::qwen35::kv_cache::HybridKvCacheSnapshot;

use super::qwen35_hybrid_persistor::{
    deserialize_hybrid_snapshot, serialize_hybrid_snapshot, FullAttnCodec,
    Qwen35HybridConfig,
};

/// Disk-backed cold-resume persistor for qwen35 hybrid snapshots.
///
/// Constructed once per `(cache_dir, config)` at engine load; the
/// `cache_dir` argument is sourced from `HF2Q_KV_PERSIST` env (iter-6).
/// Multiple `Qwen35LoadedModel` instances using the same config share a
/// fingerprint subdir (correct namespacing + zero overlap).
pub struct Qwen35DiskPersistor {
    /// Root cache directory (operator-controlled). Persisted files land
    /// under `<cache_dir>/<fingerprint>/<lcp_key>.bin`.
    cache_dir: PathBuf,
    /// Shape config; any file written by this persistor is tagged with
    /// the config's fingerprint and read-back validates against it.
    cfg: Qwen35HybridConfig,
    /// Cached fingerprint hex (16 chars). Computed once at construction
    /// from `cfg`; never re-derived per-write.
    fingerprint_hex: String,
}

impl Qwen35DiskPersistor {
    /// Construct a persistor rooted at `cache_dir` for snapshots
    /// matching `cfg`. Creates `<cache_dir>/<fingerprint>/` on disk if
    /// absent; subsequent writes land directly there.
    pub fn new(cache_dir: PathBuf, cfg: Qwen35HybridConfig) -> Result<Self> {
        let fingerprint_hex = compute_config_fingerprint_hex(&cfg);
        let dir = cache_dir.join(&fingerprint_hex);
        fs::create_dir_all(&dir)
            .with_context(|| format!("Qwen35DiskPersistor: mkdir {}", dir.display()))?;
        Ok(Self {
            cache_dir,
            cfg,
            fingerprint_hex,
        })
    }

    /// Path to the file that would back `lcp_key`, regardless of
    /// whether the file currently exists on disk.
    pub fn path_for_key(&self, lcp_key_hex: &str) -> PathBuf {
        self.cache_dir
            .join(&self.fingerprint_hex)
            .join(format!("{lcp_key_hex}.bin"))
    }

    /// Serialize `snapshot` through the QH35 envelope and write the
    /// bytes to `<cache_dir>/<fingerprint>/<lcp_key>.bin` atomically
    /// (tempfile + rename). Overwrites any existing file at the same
    /// key.
    pub fn write(
        &self,
        lcp_key_hex: &str,
        snapshot: &HybridKvCacheSnapshot,
    ) -> Result<()> {
        ensure!(
            !lcp_key_hex.is_empty(),
            "Qwen35DiskPersistor::write: empty lcp_key_hex"
        );
        ensure!(
            lcp_key_hex
                .chars()
                .all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-'),
            "Qwen35DiskPersistor::write: lcp_key_hex must be alphanumeric/_/- only \
             (got {lcp_key_hex:?})"
        );
        let bytes = serialize_hybrid_snapshot(snapshot, &self.cfg)
            .context("Qwen35DiskPersistor::write: serialize_hybrid_snapshot")?;
        let final_path = self.path_for_key(lcp_key_hex);
        let tmp_path = final_path.with_extension("bin.tmp");
        {
            let mut tmp = fs::File::create(&tmp_path).with_context(|| {
                format!("Qwen35DiskPersistor::write: create {}", tmp_path.display())
            })?;
            tmp.write_all(&bytes).with_context(|| {
                format!("Qwen35DiskPersistor::write: write_all {}", tmp_path.display())
            })?;
            tmp.sync_all().with_context(|| {
                format!("Qwen35DiskPersistor::write: sync {}", tmp_path.display())
            })?;
        }
        fs::rename(&tmp_path, &final_path).with_context(|| {
            format!(
                "Qwen35DiskPersistor::write: rename {} -> {}",
                tmp_path.display(),
                final_path.display()
            )
        })?;
        Ok(())
    }

    /// Read the snapshot keyed by `lcp_key_hex` from disk; returns
    /// `Ok(None)` when the file doesn't exist (clean cache miss).
    /// Returns `Err` on file-present-but-corrupt (caller should treat
    /// as a cache miss + delete + log; iter-6's wire-up implements that
    /// recovery).
    pub fn read(
        &self,
        lcp_key_hex: &str,
        device: &MlxDevice,
    ) -> Result<Option<HybridKvCacheSnapshot>> {
        let path = self.path_for_key(lcp_key_hex);
        let bytes = match fs::read(&path) {
            Ok(b) => b,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(None),
            Err(e) => {
                return Err(anyhow!(
                    "Qwen35DiskPersistor::read: open {}: {e}",
                    path.display()
                ))
            }
        };
        let snap = deserialize_hybrid_snapshot(&bytes, &self.cfg, device)
            .with_context(|| {
                format!(
                    "Qwen35DiskPersistor::read: deserialize {} ({} bytes)",
                    path.display(),
                    bytes.len()
                )
            })?;
        Ok(Some(snap))
    }

    /// Walk `<cache_dir>/<fingerprint>/` and return every (lcp_key,
    /// snapshot) pair that successfully deserializes. Files that fail
    /// to deserialize (corrupt / wrong-config / drift) are skipped with
    /// a warning trace — they don't poison the hydrate.
    ///
    /// Iter-6 calls this at engine load to seed the in-memory
    /// `LcpRegistry`; the caller drives the actual `LcpRegistry::insert`
    /// loop so this fn stays decoupled from the registry's borrow
    /// shape.
    pub fn hydrate_all(
        &self,
        device: &MlxDevice,
    ) -> Result<Vec<(String, HybridKvCacheSnapshot)>> {
        let dir = self.cache_dir.join(&self.fingerprint_hex);
        let entries = match fs::read_dir(&dir) {
            Ok(e) => e,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                // Empty cache dir is a clean cold start — return [].
                return Ok(Vec::new());
            }
            Err(e) => {
                return Err(anyhow!(
                    "Qwen35DiskPersistor::hydrate_all: read_dir {}: {e}",
                    dir.display()
                ))
            }
        };
        let mut out = Vec::new();
        for entry in entries {
            let entry = entry.with_context(|| {
                format!(
                    "Qwen35DiskPersistor::hydrate_all: dir entry under {}",
                    dir.display()
                )
            })?;
            let path = entry.path();
            // Skip tempfiles + non-.bin files.
            let name = match path.file_name().and_then(|s| s.to_str()) {
                Some(n) => n,
                None => continue,
            };
            if !name.ends_with(".bin") || name.ends_with(".tmp") {
                continue;
            }
            let lcp_key_hex = name.trim_end_matches(".bin").to_string();
            match self.read(&lcp_key_hex, device) {
                Ok(Some(snap)) => out.push((lcp_key_hex, snap)),
                Ok(None) => {
                    // Race window: file vanished between read_dir and
                    // open. Skip silently.
                }
                Err(e) => {
                    tracing::warn!(
                        path = %path.display(),
                        error = %e,
                        "Qwen35DiskPersistor::hydrate_all: skipping corrupt cache file"
                    );
                }
            }
        }
        Ok(out)
    }

    /// `cache_dir` getter — used by iter-6 wire-up + tests.
    pub fn cache_dir(&self) -> &Path {
        &self.cache_dir
    }

    /// Hex fingerprint of the bound config — exposed so iter-6's wire-
    /// up can log it at engine load.
    pub fn fingerprint_hex(&self) -> &str {
        &self.fingerprint_hex
    }
}

/// Compute the 8-byte (16-hex-char) fingerprint of a `Qwen35HybridConfig`.
/// Used as the namespace subdir so re-quanting / shape changes never
/// collide on disk.
///
/// Sha256 over a stable byte serialization of the shape fields; first 8
/// bytes lifted to lowercase hex.  Same input → same output across
/// processes / runs / hosts.
fn compute_config_fingerprint_hex(cfg: &Qwen35HybridConfig) -> String {
    use sha2::{Digest, Sha256};
    let mut h = Sha256::new();
    h.update(b"QH35-cfg-fp-v1");
    h.update(&cfg.n_full_attn.to_le_bytes());
    h.update(&cfg.n_linear_attn.to_le_bytes());
    h.update(&[u8::from(cfg.has_mtp)]);
    h.update(&cfg.n_seqs.to_le_bytes());
    for &dim in &cfg.full_attn_shape {
        h.update(&dim.to_le_bytes());
    }
    h.update(&[full_attn_codec_byte(cfg.full_attn_codec)]);
    for &dim in &cfg.linear_conv_shape {
        h.update(&dim.to_le_bytes());
    }
    for &dim in &cfg.linear_recurrent_shape {
        h.update(&dim.to_le_bytes());
    }
    for &dim in &cfg.mtp_shape {
        h.update(&dim.to_le_bytes());
    }
    let digest = h.finalize();
    hex::encode(&digest[..8])
}

fn full_attn_codec_byte(codec: FullAttnCodec) -> u8 {
    match codec {
        FullAttnCodec::F32Dense => 0,
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::inference::models::qwen35::kv_cache::MtpKvSnapshot;
    use mlx_native::DType;

    /// Standalone synth-snapshot helper for the disk-persistor tests
    /// (avoids cross-module test imports). Mirrors the pattern in
    /// qwen35_hybrid_persistor::tests::synth_full_attn_only_snapshot.
    fn synth_snapshot(
        device: &MlxDevice,
        cfg: &Qwen35HybridConfig,
    ) -> HybridKvCacheSnapshot {
        let elems_full: usize = cfg.full_attn_shape.iter().product::<u64>() as usize;
        let bytes_full = elems_full * std::mem::size_of::<f32>();
        let shape_full: Vec<usize> =
            cfg.full_attn_shape.iter().map(|d| *d as usize).collect();

        let mut full_attn_k = Vec::with_capacity(cfg.n_full_attn as usize);
        let mut full_attn_v = Vec::with_capacity(cfg.n_full_attn as usize);
        let mut full_attn_current_len = Vec::with_capacity(cfg.n_full_attn as usize);
        for slot in 0..cfg.n_full_attn as usize {
            let mut k = device
                .alloc_buffer(bytes_full, DType::F32, shape_full.clone())
                .unwrap();
            {
                let dst = k.as_mut_slice::<u8>().unwrap();
                for (i, b) in dst.iter_mut().enumerate() {
                    *b = ((slot * 31 + i) % 251) as u8;
                }
            }
            let mut v = device
                .alloc_buffer(bytes_full, DType::F32, shape_full.clone())
                .unwrap();
            {
                let dst = v.as_mut_slice::<u8>().unwrap();
                for (i, b) in dst.iter_mut().enumerate() {
                    *b = ((slot * 37 + i) % 251) as u8;
                }
            }
            full_attn_k.push(k);
            full_attn_v.push(v);
            full_attn_current_len.push((0..cfg.n_seqs).map(|s| (slot as u32) + s).collect());
        }

        let elems_conv: usize = cfg.linear_conv_shape.iter().product::<u64>() as usize;
        let bytes_conv = elems_conv * std::mem::size_of::<f32>();
        let shape_conv: Vec<usize> =
            cfg.linear_conv_shape.iter().map(|d| *d as usize).collect();
        let elems_rec: usize = cfg.linear_recurrent_shape.iter().product::<u64>() as usize;
        let bytes_rec = elems_rec * std::mem::size_of::<f32>();
        let shape_rec: Vec<usize> = cfg
            .linear_recurrent_shape
            .iter()
            .map(|d| *d as usize)
            .collect();

        let mut linear_conv = Vec::new();
        let mut linear_recurrent = Vec::new();
        for slot in 0..cfg.n_linear_attn as usize {
            let mut c = device
                .alloc_buffer(bytes_conv, DType::F32, shape_conv.clone())
                .unwrap();
            {
                let dst = c.as_mut_slice::<u8>().unwrap();
                for (i, b) in dst.iter_mut().enumerate() {
                    *b = ((slot * 41 + i) % 251) as u8;
                }
            }
            let mut r = device
                .alloc_buffer(bytes_rec, DType::F32, shape_rec.clone())
                .unwrap();
            {
                let dst = r.as_mut_slice::<u8>().unwrap();
                for (i, b) in dst.iter_mut().enumerate() {
                    *b = ((slot * 43 + i) % 251) as u8;
                }
            }
            linear_conv.push(c);
            linear_recurrent.push(r);
        }

        let mtp = if cfg.has_mtp {
            let elems: usize = cfg.mtp_shape.iter().product::<u64>() as usize;
            let bytes_len = elems * std::mem::size_of::<f32>();
            let shape: Vec<usize> = cfg.mtp_shape.iter().map(|d| *d as usize).collect();
            let mut k = device
                .alloc_buffer(bytes_len, DType::F32, shape.clone())
                .unwrap();
            {
                let dst = k.as_mut_slice::<u8>().unwrap();
                for (i, b) in dst.iter_mut().enumerate() {
                    *b = ((47 * i + 11) % 251) as u8;
                }
            }
            let mut v = device.alloc_buffer(bytes_len, DType::F32, shape).unwrap();
            {
                let dst = v.as_mut_slice::<u8>().unwrap();
                for (i, b) in dst.iter_mut().enumerate() {
                    *b = ((53 * i + 13) % 251) as u8;
                }
            }
            Some(MtpKvSnapshot {
                k,
                v,
                current_len: (0..cfg.n_seqs).map(|s| 100 + s).collect(),
            })
        } else {
            None
        };

        HybridKvCacheSnapshot {
            full_attn_k,
            full_attn_v,
            full_attn_current_len,
            mtp,
            linear_conv,
            linear_recurrent,
        }
    }

    fn synth_cfg() -> Qwen35HybridConfig {
        Qwen35HybridConfig {
            n_full_attn: 2,
            n_linear_attn: 3,
            has_mtp: true,
            n_seqs: 1,
            full_attn_shape: [1, 2, 8, 4],
            full_attn_codec: FullAttnCodec::F32Dense,
            linear_conv_shape: [4, 3, 1],
            linear_recurrent_shape: [4, 8, 2, 1],
            mtp_shape: [1, 4, 8, 4],
        }
    }

    fn snapshots_byte_equal(a: &HybridKvCacheSnapshot, b: &HybridKvCacheSnapshot) -> bool {
        if a.full_attn_k.len() != b.full_attn_k.len() {
            return false;
        }
        if a.full_attn_v.len() != b.full_attn_v.len() {
            return false;
        }
        if a.full_attn_current_len != b.full_attn_current_len {
            return false;
        }
        for i in 0..a.full_attn_k.len() {
            if a.full_attn_k[i].as_slice::<u8>().unwrap()
                != b.full_attn_k[i].as_slice::<u8>().unwrap()
            {
                return false;
            }
            if a.full_attn_v[i].as_slice::<u8>().unwrap()
                != b.full_attn_v[i].as_slice::<u8>().unwrap()
            {
                return false;
            }
        }
        if a.linear_conv.len() != b.linear_conv.len() {
            return false;
        }
        for i in 0..a.linear_conv.len() {
            if a.linear_conv[i].as_slice::<u8>().unwrap()
                != b.linear_conv[i].as_slice::<u8>().unwrap()
            {
                return false;
            }
            if a.linear_recurrent[i].as_slice::<u8>().unwrap()
                != b.linear_recurrent[i].as_slice::<u8>().unwrap()
            {
                return false;
            }
        }
        match (&a.mtp, &b.mtp) {
            (None, None) => true,
            (Some(am), Some(bm)) => {
                am.k.as_slice::<u8>().unwrap() == bm.k.as_slice::<u8>().unwrap()
                    && am.v.as_slice::<u8>().unwrap() == bm.v.as_slice::<u8>().unwrap()
                    && am.current_len == bm.current_len
            }
            _ => false,
        }
    }

    #[test]
    fn qh35_disk_round_trip_via_tempdir() {
        let device = match MlxDevice::new() {
            Ok(d) => d,
            Err(e) => {
                eprintln!("skipping: no Metal device: {e}");
                return;
            }
        };
        let cfg = synth_cfg();
        let tempdir = std::env::temp_dir().join(format!(
            "qh35-disk-test-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or(0)
        ));
        let _ = fs::remove_dir_all(&tempdir);
        let persistor = Qwen35DiskPersistor::new(tempdir.clone(), cfg.clone()).unwrap();
        let snap = synth_snapshot(&device, &cfg);
        persistor.write("deadbeef_cafebabe", &snap).unwrap();
        let restored = persistor
            .read("deadbeef_cafebabe", &device)
            .unwrap()
            .expect("write+read should round-trip");
        assert!(snapshots_byte_equal(&snap, &restored));
        let _ = fs::remove_dir_all(&tempdir);
    }

    #[test]
    fn qh35_disk_read_missing_returns_none() {
        let device = match MlxDevice::new() {
            Ok(d) => d,
            Err(e) => {
                eprintln!("skipping: no Metal device: {e}");
                return;
            }
        };
        let cfg = synth_cfg();
        let tempdir = std::env::temp_dir().join(format!(
            "qh35-disk-miss-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or(0)
        ));
        let _ = fs::remove_dir_all(&tempdir);
        let persistor = Qwen35DiskPersistor::new(tempdir.clone(), cfg).unwrap();
        let result = persistor.read("nonexistent_key", &device).unwrap();
        assert!(result.is_none());
        let _ = fs::remove_dir_all(&tempdir);
    }

    #[test]
    fn qh35_disk_hydrate_all_collects_every_file() {
        let device = match MlxDevice::new() {
            Ok(d) => d,
            Err(e) => {
                eprintln!("skipping: no Metal device: {e}");
                return;
            }
        };
        let cfg = synth_cfg();
        let tempdir = std::env::temp_dir().join(format!(
            "qh35-disk-hydrate-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or(0)
        ));
        let _ = fs::remove_dir_all(&tempdir);
        let persistor = Qwen35DiskPersistor::new(tempdir.clone(), cfg.clone()).unwrap();
        let snap = synth_snapshot(&device, &cfg);
        let keys = ["key_a", "key_b", "key_c"];
        for k in &keys {
            persistor.write(k, &snap).unwrap();
        }
        let mut hydrated = persistor.hydrate_all(&device).unwrap();
        hydrated.sort_by(|a, b| a.0.cmp(&b.0));
        assert_eq!(hydrated.len(), keys.len());
        for (got, expected) in hydrated.iter().zip(keys.iter()) {
            assert_eq!(&got.0, expected);
            assert!(snapshots_byte_equal(&snap, &got.1));
        }
        let _ = fs::remove_dir_all(&tempdir);
    }

    #[test]
    fn qh35_disk_fingerprint_subdir_isolates_cfg_drift() {
        // Two configs with different shapes get different fingerprint
        // subdirs — a key written under cfg-A is invisible under cfg-B.
        let device = match MlxDevice::new() {
            Ok(d) => d,
            Err(e) => {
                eprintln!("skipping: no Metal device: {e}");
                return;
            }
        };
        let mut cfg_a = synth_cfg();
        let mut cfg_b = synth_cfg();
        cfg_b.full_attn_shape = [1, 4, 8, 4]; // n_kv_heads=4 instead of 2
        let tempdir = std::env::temp_dir().join(format!(
            "qh35-disk-fp-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or(0)
        ));
        let _ = fs::remove_dir_all(&tempdir);
        // Make cfg_a + cfg_b structurally distinct (avoid any chance of
        // false equality between random byte patterns at the same key).
        cfg_a.n_seqs = 1;
        cfg_b.n_seqs = 1;
        let pa = Qwen35DiskPersistor::new(tempdir.clone(), cfg_a.clone()).unwrap();
        let pb = Qwen35DiskPersistor::new(tempdir.clone(), cfg_b.clone()).unwrap();
        assert_ne!(
            pa.fingerprint_hex(),
            pb.fingerprint_hex(),
            "differently-shaped configs MUST produce different fingerprints"
        );
        let snap_a = synth_snapshot(&device, &cfg_a);
        pa.write("shared_key", &snap_a).unwrap();
        // pb under cfg_b should NOT see pa's file (different subdir).
        let result_b = pb.read("shared_key", &device).unwrap();
        assert!(
            result_b.is_none(),
            "cfg_b persistor must not see cfg_a's snapshot under shared key"
        );
        let _ = fs::remove_dir_all(&tempdir);
    }

    #[test]
    fn qh35_disk_atomic_overwrite_replaces_prior_content() {
        let device = match MlxDevice::new() {
            Ok(d) => d,
            Err(e) => {
                eprintln!("skipping: no Metal device: {e}");
                return;
            }
        };
        let cfg = synth_cfg();
        let tempdir = std::env::temp_dir().join(format!(
            "qh35-disk-overwrite-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or(0)
        ));
        let _ = fs::remove_dir_all(&tempdir);
        let persistor = Qwen35DiskPersistor::new(tempdir.clone(), cfg.clone()).unwrap();
        // Write snap-A, then overwrite with snap-B (different patterns).
        let snap_a = synth_snapshot(&device, &cfg);
        persistor.write("the_key", &snap_a).unwrap();
        // snap_b matches cfg (same persistor / same fingerprint subdir);
        // we mutate its bytes after construction so the on-disk content
        // demonstrably overwrites snap_a's content.
        let mut snap_b = synth_snapshot(&device, &cfg);
        // Mutate snap_b's first full-attn K bytes so it byte-differs.
        {
            let dst = snap_b.full_attn_k[0].as_mut_slice::<u8>().unwrap();
            for b in dst.iter_mut() {
                *b = b.wrapping_add(7);
            }
        }
        persistor.write("the_key", &snap_b).unwrap();
        let restored = persistor.read("the_key", &device).unwrap().unwrap();
        assert!(snapshots_byte_equal(&snap_b, &restored));
        assert!(!snapshots_byte_equal(&snap_a, &restored));
        let _ = fs::remove_dir_all(&tempdir);
    }
}
