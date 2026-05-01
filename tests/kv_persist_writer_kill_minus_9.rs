//! ADR-017 §A.2 integration test — `kill -9` mid-write atomicity proof.
//!
//! Forks a child that writes 10× 1-MiB blocks via the production
//! `DiskBlockStore` + `AsyncWriterHandle`, then SIGKILLs it mid-stream.
//! Asserts `<sha>.safetensors` files at canonical names parse cleanly
//! (atomic rename held) and tolerates `*.tmp.<pid>` orphans (§D5+§D8).

#![cfg(unix)]

use hf2q::serve::kv_persist::{
    block_store::{DiskBlockStore, WriteJob},
    format::{self, compute_model_fingerprint, BlockHash, EnvelopeHeader, ModelFingerprint, ParentBlockHash, BLOCK_TOKENS, CURRENT_FORMAT_VERSION},
    writer::AsyncWriterHandle,
};
use sha2::{Digest, Sha256};
use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant};

fn fp() -> ModelFingerprint {
    compute_model_fingerprint("kill9/test", "Q4_0", "hf2q-1.0", "deadbeef", "tpl")
}

fn make_block(seed: u32) -> (Vec<u8>, EnvelopeHeader) {
    let body: Vec<u8> = (0..262_144u32).flat_map(|i| (i.wrapping_add(seed)).to_le_bytes()).collect();
    let bh: [u8; 32] = Sha256::digest(&body).into();
    (body, EnvelopeHeader {
        format_version: CURRENT_FORMAT_VERSION.0, model_fingerprint: fp(),
        block_hash: BlockHash(bh), parent_block_hash: ParentBlockHash(None),
        payload_kind: "kv-kill9".into(), codec_version: 1, n_tokens: BLOCK_TOKENS,
    })
}

fn child_body(cache_root: &std::path::Path) -> ! {
    let store = Arc::new(DiskBlockStore::new(cache_root.to_path_buf(), 0).expect("store"));
    let handle = AsyncWriterHandle::spawn(Arc::clone(&store), 4);
    for s in 0u32..10 {
        let (body, header) = make_block(s);
        let _ = handle.enqueue_blocking(WriteJob { header, body, completion_tx: None });
    }
    let deadline = Instant::now() + Duration::from_secs(5);
    while Instant::now() < deadline { std::thread::sleep(Duration::from_millis(10)); }
    std::process::exit(0)
}

#[test]
fn kill_minus_9_mid_write_leaves_committed_blocks_and_no_partial_named_files() {
    let nanos = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos();
    let cache_root = std::env::temp_dir().join(format!("hf2q-kv-kill9-{}-{nanos}", std::process::id()));
    std::fs::create_dir_all(&cache_root).expect("mkdir");

    // SAFETY: pre-fork we are single-threaded; post-fork the child
    // spawns its writer thread on its own main thread.
    let pid = unsafe { libc::fork() };
    assert!(pid >= 0, "fork: {}", std::io::Error::last_os_error());
    if pid == 0 { child_body(&cache_root); }

    std::thread::sleep(Duration::from_millis(30));
    assert_eq!(unsafe { libc::kill(pid, libc::SIGKILL) }, 0, "kill: {}", std::io::Error::last_os_error());
    let mut status: libc::c_int = 0;
    assert_eq!(unsafe { libc::waitpid(pid, &mut status as *mut _, 0) }, pid, "waitpid");

    let kv_dir = cache_root.join("models").join(fp().short_hex()).join("kv");
    if !kv_dir.exists() { eprintln!("[kv-kill9] kill before any write; skip"); let _ = std::fs::remove_dir_all(&cache_root); return; }

    let (mut final_files, mut tmp_files): (Vec<PathBuf>, Vec<PathBuf>) = (Vec::new(), Vec::new());
    for fanout_ent in std::fs::read_dir(&kv_dir).unwrap().filter_map(|e| e.ok()) {
        if !fanout_ent.path().is_dir() { continue; }
        for ent in std::fs::read_dir(fanout_ent.path()).unwrap().filter_map(|e| e.ok()) {
            let p = ent.path();
            if !p.is_file() { continue; }
            let n = p.file_name().unwrap().to_string_lossy().into_owned();
            if n.contains(".tmp.") { tmp_files.push(p); } else { final_files.push(p); }
        }
    }

    if final_files.is_empty() {
        eprintln!("[kv-kill9] kill before any rename committed ({} tmp); skip", tmp_files.len());
        let _ = std::fs::remove_dir_all(&cache_root); return;
    }
    for p in &final_files {
        let (h, body) = format::read_envelope_body(p).unwrap_or_else(|e| panic!("partial-named-final {p:?}: {e}"));
        assert_eq!(h.format_version, CURRENT_FORMAT_VERSION.0);
        assert!(!body.is_empty());
    }
    eprintln!("[kv-kill9] PASS — {} committed, {} tmp orphans, 0 partial-named-final",
        final_files.len(), tmp_files.len());
    let _ = std::fs::remove_dir_all(&cache_root);
}
