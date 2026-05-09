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
/// Constructed once per `cache_dir` at engine load; the `cache_dir`
/// argument is sourced from `HF2Q_KV_PERSIST` env (iter-6).
/// `Qwen35HybridConfig` is passed PER-CALL on `write` / `read` /
/// `hydrate_for_cfg` because cache shape (`max_seq_len`, `n_seqs`)
/// is per-prefill, not load-time (kv_cache.rs:347+ — see ADR-027
/// §4.7 iter-6a finding).
///
/// Multiple `Qwen35LoadedModel` instances using the same `cache_dir`
/// can safely share one persistor; per-cfg isolation is enforced by
/// the fingerprint subdir layout
/// (`<cache_dir>/<cfg-fingerprint>/<lcp_key>.bin`) — cfg drift on
/// any shape field re-routes to a different subdir, so re-quanting
/// or different `max_seq_len` requests cannot collide.
pub struct Qwen35DiskPersistor {
    /// Root cache directory (operator-controlled). Persisted files land
    /// under `<cache_dir>/<cfg-fingerprint>/<lcp_key>.bin`.
    cache_dir: PathBuf,
}

impl Qwen35DiskPersistor {
    /// Construct a persistor rooted at `cache_dir`. Does NOT create
    /// the directory yet — the per-cfg subdir is created lazily on
    /// the first `write` for that cfg.
    pub fn new(cache_dir: PathBuf) -> Result<Self> {
        // Validate the parent dir is creatable; do not pre-create
        // per-cfg subdirs (we don't know which cfgs will be used yet).
        fs::create_dir_all(&cache_dir).with_context(|| {
            format!("Qwen35DiskPersistor: mkdir {}", cache_dir.display())
        })?;
        Ok(Self { cache_dir })
    }

    /// Path to the file that would back `(cfg, lcp_key)`, regardless
    /// of whether the file currently exists on disk.
    pub fn path_for_key(&self, cfg: &Qwen35HybridConfig, lcp_key_hex: &str) -> PathBuf {
        let fingerprint_hex = compute_config_fingerprint_hex(cfg);
        self.cache_dir
            .join(fingerprint_hex)
            .join(format!("{lcp_key_hex}.bin"))
    }

    /// Serialize `snapshot` through the QH35 envelope (under `cfg`)
    /// and write the bytes to
    /// `<cache_dir>/<cfg-fingerprint>/<lcp_key>.bin` atomically
    /// (tempfile + rename). Creates the per-cfg subdir on first use.
    /// Overwrites any existing file at the same key.
    pub fn write(
        &self,
        cfg: &Qwen35HybridConfig,
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
        let bytes = serialize_hybrid_snapshot(snapshot, cfg)
            .context("Qwen35DiskPersistor::write: serialize_hybrid_snapshot")?;
        let final_path = self.path_for_key(cfg, lcp_key_hex);
        let cfg_dir = final_path
            .parent()
            .ok_or_else(|| anyhow!("Qwen35DiskPersistor::write: path has no parent"))?;
        fs::create_dir_all(cfg_dir).with_context(|| {
            format!(
                "Qwen35DiskPersistor::write: per-cfg mkdir {}",
                cfg_dir.display()
            )
        })?;
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

    /// Read the snapshot keyed by `(cfg, lcp_key_hex)` from disk;
    /// returns `Ok(None)` when the file doesn't exist (clean cache
    /// miss). Returns `Err` on file-present-but-corrupt (caller
    /// should treat as a cache miss + delete + log; iter-6b's
    /// wire-up implements that recovery).
    pub fn read(
        &self,
        cfg: &Qwen35HybridConfig,
        lcp_key_hex: &str,
        device: &MlxDevice,
    ) -> Result<Option<HybridKvCacheSnapshot>> {
        let path = self.path_for_key(cfg, lcp_key_hex);
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
        let snap = deserialize_hybrid_snapshot(&bytes, cfg, device).with_context(|| {
            format!(
                "Qwen35DiskPersistor::read: deserialize {} ({} bytes)",
                path.display(),
                bytes.len()
            )
        })?;
        Ok(Some(snap))
    }

    /// Walk `<cache_dir>/<cfg-fingerprint>/` and return every
    /// `(lcp_key, snapshot)` pair that successfully deserializes
    /// against `cfg`. Files that fail to deserialize (corrupt /
    /// wrong-config / drift) are skipped with a warning trace — they
    /// don't poison the hydrate.
    ///
    /// Iter-6b calls this at first prefill (when cfg is known) to
    /// seed the in-memory `LcpRegistry` for that cfg. Earlier-cfg
    /// snapshots (under different fingerprint subdirs) are NOT
    /// hydrated — they live correctly under their own subdir until a
    /// matching-cfg prefill arrives.
    pub fn hydrate_for_cfg(
        &self,
        cfg: &Qwen35HybridConfig,
        device: &MlxDevice,
    ) -> Result<Vec<(String, HybridKvCacheSnapshot)>> {
        let fingerprint_hex = compute_config_fingerprint_hex(cfg);
        let dir = self.cache_dir.join(&fingerprint_hex);
        let entries = match fs::read_dir(&dir) {
            Ok(e) => e,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                // Empty cfg-subdir is a clean cold start — return [].
                return Ok(Vec::new());
            }
            Err(e) => {
                return Err(anyhow!(
                    "Qwen35DiskPersistor::hydrate_for_cfg: read_dir {}: {e}",
                    dir.display()
                ))
            }
        };
        let mut out = Vec::new();
        for entry in entries {
            let entry = entry.with_context(|| {
                format!(
                    "Qwen35DiskPersistor::hydrate_for_cfg: dir entry under {}",
                    dir.display()
                )
            })?;
            let path = entry.path();
            let name = match path.file_name().and_then(|s| s.to_str()) {
                Some(n) => n,
                None => continue,
            };
            if !name.ends_with(".bin") || name.ends_with(".tmp") {
                continue;
            }
            let lcp_key_hex = name.trim_end_matches(".bin").to_string();
            match self.read(cfg, &lcp_key_hex, device) {
                Ok(Some(snap)) => out.push((lcp_key_hex, snap)),
                Ok(None) => {
                    // Race window: file vanished between read_dir and
                    // open. Skip silently.
                }
                Err(e) => {
                    tracing::warn!(
                        path = %path.display(),
                        error = %e,
                        "Qwen35DiskPersistor::hydrate_for_cfg: skipping corrupt cache file"
                    );
                }
            }
        }
        Ok(out)
    }

    /// `cache_dir` getter — used by iter-6b wire-up + tests.
    pub fn cache_dir(&self) -> &Path {
        &self.cache_dir
    }

    /// Compute the cfg-fingerprint subdir name (16-hex chars) for an
    /// arbitrary `cfg` — exposed for tests + log lines that want to
    /// surface "which cfg-subdir is in use" without exposing the full
    /// hashing details.
    pub fn fingerprint_hex_for(cfg: &Qwen35HybridConfig) -> String {
        compute_config_fingerprint_hex(cfg)
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

    /// Make a unique tempdir per test so parallel test execution
    /// doesn't collide.
    fn unique_tempdir(label: &str) -> PathBuf {
        std::env::temp_dir().join(format!(
            "qh35-disk-{label}-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or(0)
        ))
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
        let tempdir = unique_tempdir("rt");
        let _ = fs::remove_dir_all(&tempdir);
        let persistor = Qwen35DiskPersistor::new(tempdir.clone()).unwrap();
        let snap = synth_snapshot(&device, &cfg);
        persistor.write(&cfg, "deadbeef_cafebabe", &snap).unwrap();
        let restored = persistor
            .read(&cfg, "deadbeef_cafebabe", &device)
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
        let tempdir = unique_tempdir("miss");
        let _ = fs::remove_dir_all(&tempdir);
        let persistor = Qwen35DiskPersistor::new(tempdir.clone()).unwrap();
        let result = persistor.read(&cfg, "nonexistent_key", &device).unwrap();
        assert!(result.is_none());
        let _ = fs::remove_dir_all(&tempdir);
    }

    #[test]
    fn qh35_disk_hydrate_for_cfg_collects_every_file() {
        let device = match MlxDevice::new() {
            Ok(d) => d,
            Err(e) => {
                eprintln!("skipping: no Metal device: {e}");
                return;
            }
        };
        let cfg = synth_cfg();
        let tempdir = unique_tempdir("hydrate");
        let _ = fs::remove_dir_all(&tempdir);
        let persistor = Qwen35DiskPersistor::new(tempdir.clone()).unwrap();
        let snap = synth_snapshot(&device, &cfg);
        let keys = ["key_a", "key_b", "key_c"];
        for k in &keys {
            persistor.write(&cfg, k, &snap).unwrap();
        }
        let mut hydrated = persistor.hydrate_for_cfg(&cfg, &device).unwrap();
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
        // Two configs with different shapes routed to different
        // fingerprint subdirs by ONE shared persistor — a key written
        // under cfg-A must NOT appear under cfg-B.
        let device = match MlxDevice::new() {
            Ok(d) => d,
            Err(e) => {
                eprintln!("skipping: no Metal device: {e}");
                return;
            }
        };
        let cfg_a = synth_cfg();
        let mut cfg_b = synth_cfg();
        cfg_b.full_attn_shape = [1, 4, 8, 4]; // n_kv_heads=4 instead of 2
        let tempdir = unique_tempdir("fp");
        let _ = fs::remove_dir_all(&tempdir);
        let persistor = Qwen35DiskPersistor::new(tempdir.clone()).unwrap();
        assert_ne!(
            Qwen35DiskPersistor::fingerprint_hex_for(&cfg_a),
            Qwen35DiskPersistor::fingerprint_hex_for(&cfg_b),
            "differently-shaped configs MUST produce different fingerprints"
        );
        let snap_a = synth_snapshot(&device, &cfg_a);
        persistor.write(&cfg_a, "shared_key", &snap_a).unwrap();
        // Reading the SAME key under cfg_b targets a different subdir
        // → must miss.
        let result_b = persistor.read(&cfg_b, "shared_key", &device).unwrap();
        assert!(
            result_b.is_none(),
            "cfg_b read must not see cfg_a's snapshot under shared key"
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
        let tempdir = unique_tempdir("overwrite");
        let _ = fs::remove_dir_all(&tempdir);
        let persistor = Qwen35DiskPersistor::new(tempdir.clone()).unwrap();
        // Write snap-A, then overwrite with mutated snap-B.
        let snap_a = synth_snapshot(&device, &cfg);
        persistor.write(&cfg, "the_key", &snap_a).unwrap();
        let mut snap_b = synth_snapshot(&device, &cfg);
        {
            let dst = snap_b.full_attn_k[0].as_mut_slice::<u8>().unwrap();
            for b in dst.iter_mut() {
                *b = b.wrapping_add(7);
            }
        }
        persistor.write(&cfg, "the_key", &snap_b).unwrap();
        let restored = persistor.read(&cfg, "the_key", &device).unwrap().unwrap();
        assert!(snapshots_byte_equal(&snap_b, &restored));
        assert!(!snapshots_byte_equal(&snap_a, &restored));
        let _ = fs::remove_dir_all(&tempdir);
    }

    #[test]
    fn qh35_disk_multi_cfg_cohabit_one_persistor() {
        // Iter-6b API affordance: a single persistor handles multiple
        // cfgs (e.g., two prefills with different max_seq_len). Each
        // cfg lives in its own fingerprint subdir; reads + writes
        // route correctly per cfg.
        let device = match MlxDevice::new() {
            Ok(d) => d,
            Err(e) => {
                eprintln!("skipping: no Metal device: {e}");
                return;
            }
        };
        let cfg_a = synth_cfg();
        // cfg_b: distinct full_attn_shape (different max_seq_len would
        // be the production drift; we use a different head count so
        // synth_snapshot produces a structurally-different cache).
        let mut cfg_b = synth_cfg();
        cfg_b.full_attn_shape = [1, 4, 8, 4];
        let tempdir = unique_tempdir("multi-cfg");
        let _ = fs::remove_dir_all(&tempdir);
        let persistor = Qwen35DiskPersistor::new(tempdir.clone()).unwrap();
        let snap_a = synth_snapshot(&device, &cfg_a);
        let snap_b = synth_snapshot(&device, &cfg_b);
        persistor.write(&cfg_a, "key1", &snap_a).unwrap();
        persistor.write(&cfg_b, "key1", &snap_b).unwrap();
        // Each cfg sees ITS OWN snap at the same key.
        let r_a = persistor.read(&cfg_a, "key1", &device).unwrap().unwrap();
        let r_b = persistor.read(&cfg_b, "key1", &device).unwrap().unwrap();
        assert!(snapshots_byte_equal(&snap_a, &r_a));
        assert!(snapshots_byte_equal(&snap_b, &r_b));
        // Hydrate-for-cfg returns ONLY that cfg's entries.
        let h_a = persistor.hydrate_for_cfg(&cfg_a, &device).unwrap();
        let h_b = persistor.hydrate_for_cfg(&cfg_b, &device).unwrap();
        assert_eq!(h_a.len(), 1);
        assert_eq!(h_b.len(), 1);
        let _ = fs::remove_dir_all(&tempdir);
    }
}
