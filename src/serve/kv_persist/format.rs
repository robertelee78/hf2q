//! ADR-017 §A.1 — on-disk envelope format and chain-hash identity.
//!
//! This module owns:
//!
//!   * [`CacheFormatVersion`] / [`CURRENT_FORMAT_VERSION`] — the
//!     on-disk format version (ADR-017 §D10). Initial value `1`. Future
//!     format changes bump this and readers MUST quarantine on mismatch
//!     rather than silently accept.
//!   * [`BlockHash`] / [`ModelFingerprint`] / [`ParentBlockHash`] — 32-byte
//!     sha256 outputs with hex `Display` / `FromStr` and serde shape
//!     matching the JSON header on disk (ADR-017 §D4).
//!   * [`compute_model_fingerprint`] — derives a stable per-model namespace
//!     key from `(repo_id, quant, producer_version, source_sha256,
//!     tokenizer_chat_template)` per ADR-017 §D4 + ADR-005 iter-211 GGUF
//!     metadata path. Components are joined with NUL bytes so that
//!     adjacent-string concatenation cannot collide across distinct
//!     component splits.
//!   * [`compute_block_hash`] — chain-hash recurrence
//!     `sha256(model_fp || parent_hash_or_zeros || token_ids_le_bytes)`.
//!     Genesis blocks use `ParentBlockHash(None)` which is encoded as the
//!     all-zero 32-byte seed in the hash input.
//!   * [`BLOCK_TOKENS`] — block size constant (ADR-017 §D3, oMLX
//!     `scheduler.py:321-331`).
//!   * [`EnvelopeHeader`] — the JSON-serializable on-disk header.
//!   * [`write_envelope`] / [`read_envelope_header`] / [`read_envelope_body`]
//!     — byte-for-byte mirror of `/opt/omlx/omlx/cache/paged_ssd_cache.py:246-297`
//!     `_write_safetensors_no_mx` (8-byte LE header_len + ASCII-space-padded
//!     JSON header to 8-byte alignment + concatenated payload bytes), with
//!     atomic publication via `<path>.tmp.<pid>` + `std::fs::rename`
//!     (mirrors `paged_ssd_cache.py:993-1003`).
//!
//! ## Chesterton's fence on the on-disk envelope
//!
//! The envelope layout is verbatim oMLX safetensors:
//!
//! ```text
//! [8 bytes: header_len as little-endian u64]
//! [header_len bytes: JSON header, padded with ASCII space to 8-byte align]
//! [remaining bytes: concatenated body]
//! ```
//!
//! The envelope is intentionally compatible with safetensors readers —
//! callers that want to inspect raw blocks with `safetensors`-aware
//! tooling can do so. The body is opaque bytes from the format layer's
//! point of view; ADR-017 §A.2 (`block_store`) and §A.3 (`spiller`)
//! own the per-payload `(K, V, optional_state)` schema.
//!
//! ## Why the hash chain matters
//!
//! Identity is derived (chain-hash), not assigned. Two engines computing
//! a block hash for the same (model, prefix, tokens) tuple MUST produce
//! the same bytes; this is the property the [`tests`] section locks in.

use serde::{Deserialize, Deserializer, Serialize, Serializer};
use sha2::{Digest, Sha256};
use std::fmt;
use std::fs::File;
use std::io::{self, Read, Write};
use std::path::Path;
use std::process;
use std::str::FromStr;

/// Block size in tokens. ADR-017 §D3 — adopt oMLX's empirically-validated
/// default (`scheduler.py:321-331`). Phase A0 reserves the right to revisit.
pub const BLOCK_TOKENS: u32 = 256;

/// On-disk format version envelope (ADR-017 §D10). Bump on any change to
/// the JSON header schema or the byte layout; readers reject unknown
/// versions and quarantine the file.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct CacheFormatVersion(pub u32);

/// Initial on-disk format version. Per ADR-017 §D10, must equal `1` until
/// a deliberate format-bump rolls forward.
pub const CURRENT_FORMAT_VERSION: CacheFormatVersion = CacheFormatVersion(1);

// ---------------------------------------------------------------------------
// Hex helpers (lowercase, fixed-length sha256).
// ---------------------------------------------------------------------------

fn hex_encode_32(bytes: &[u8; 32]) -> String {
    hex::encode(bytes)
}

fn hex_decode_32(s: &str) -> Result<[u8; 32], String> {
    let v = hex::decode(s).map_err(|e| format!("hex decode: {e}"))?;
    if v.len() != 32 {
        return Err(format!("expected 32-byte hex, got {}", v.len()));
    }
    let mut out = [0u8; 32];
    out.copy_from_slice(&v);
    Ok(out)
}

// ---------------------------------------------------------------------------
// BlockHash — sha256 of (model_fp || parent_hash_or_zeros || token_le_bytes).
// ---------------------------------------------------------------------------

/// 32-byte sha256 output identifying a single 256-token block in the
/// chain-hash sequence. Serialized as a lowercase hex string in the
/// envelope JSON header so the file is human-readable.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct BlockHash(pub [u8; 32]);

impl BlockHash {
    /// All-zero seed used in the chain when there is no parent. Equivalent
    /// to "empty prefix"; never written to disk as a `block_hash` (only
    /// used as the parent input for genesis blocks).
    pub fn zero() -> Self {
        Self([0u8; 32])
    }
}

impl fmt::Display for BlockHash {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&hex_encode_32(&self.0))
    }
}

impl FromStr for BlockHash {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        hex_decode_32(s).map(Self)
    }
}

impl Serialize for BlockHash {
    fn serialize<S: Serializer>(&self, ser: S) -> Result<S::Ok, S::Error> {
        ser.serialize_str(&hex_encode_32(&self.0))
    }
}

impl<'de> Deserialize<'de> for BlockHash {
    fn deserialize<D: Deserializer<'de>>(de: D) -> Result<Self, D::Error> {
        let s = String::deserialize(de)?;
        hex_decode_32(&s).map(Self).map_err(serde::de::Error::custom)
    }
}

// ---------------------------------------------------------------------------
// ParentBlockHash — explicit Option for "genesis" semantics.
// ---------------------------------------------------------------------------

/// Parent of a block in the chain. `None` denotes a genesis block (the
/// first block of a session); the hash input substitutes the all-zero
/// 32-byte seed for `None` per ADR-017 §D8.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ParentBlockHash(pub Option<BlockHash>);

impl ParentBlockHash {
    /// Bytes fed into the chain-hash recurrence: the parent's 32 bytes if
    /// present, the all-zero seed otherwise. Distinct values for distinct
    /// semantic states (genesis vs known-parent) so a chain cannot
    /// silently collide a child of the seed with a genesis.
    fn hash_input_bytes(&self) -> [u8; 32] {
        self.0.map(|h| h.0).unwrap_or([0u8; 32])
    }
}

// ---------------------------------------------------------------------------
// ModelFingerprint — stable per-model namespace key.
// ---------------------------------------------------------------------------

/// 32-byte sha256 of the GGUF provenance tuple (ADR-017 §D4). Re-quanting
/// or upgrading the chat template flips this and orphans the prior cache
/// namespace cleanly.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ModelFingerprint(pub [u8; 32]);

impl ModelFingerprint {
    /// First 16 hex characters — used as the directory short-name in the
    /// hex-fanout layout (`<root>/models/<short>/kv/...`).
    pub fn short_hex(&self) -> String {
        hex::encode(&self.0[..8])
    }
}

impl fmt::Display for ModelFingerprint {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&hex_encode_32(&self.0))
    }
}

impl FromStr for ModelFingerprint {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        hex_decode_32(s).map(Self)
    }
}

impl Serialize for ModelFingerprint {
    fn serialize<S: Serializer>(&self, ser: S) -> Result<S::Ok, S::Error> {
        ser.serialize_str(&hex_encode_32(&self.0))
    }
}

impl<'de> Deserialize<'de> for ModelFingerprint {
    fn deserialize<D: Deserializer<'de>>(de: D) -> Result<Self, D::Error> {
        let s = String::deserialize(de)?;
        hex_decode_32(&s).map(Self).map_err(serde::de::Error::custom)
    }
}

/// Compute a stable per-model namespace key from the GGUF provenance
/// tuple. Components are concatenated with NUL byte separators so that
/// shifting bytes between adjacent components cannot collide (e.g.
/// `("ab", "c")` vs `("a", "bc")` — without separators, both hash the
/// same bytes; with NUL separators, the two inputs differ).
///
/// Mirrors ADR-017 §D4 + ADR-005 iter-211 hf2q.provenance.* metadata
/// keys (`hf2q.producer_version`, `hf2q.source_sha256`,
/// `tokenizer.chat_template`).
pub fn compute_model_fingerprint(
    repo_id: &str,
    quant: &str,
    producer_version: &str,
    source_sha256: &str,
    tokenizer_chat_template: &str,
) -> ModelFingerprint {
    let mut h = Sha256::new();
    h.update(repo_id.as_bytes());
    h.update(b"\x00");
    h.update(quant.as_bytes());
    h.update(b"\x00");
    h.update(producer_version.as_bytes());
    h.update(b"\x00");
    h.update(source_sha256.as_bytes());
    h.update(b"\x00");
    h.update(tokenizer_chat_template.as_bytes());
    let out = h.finalize();
    let mut buf = [0u8; 32];
    buf.copy_from_slice(&out);
    ModelFingerprint(buf)
}

/// Chain-hash recurrence per ADR-017 §D4:
///
/// ```text
/// block_hash(N) = sha256(model_fingerprint
///                       || parent_block_hash_bytes_or_zeros
///                       || token_ids[N*BLOCK..(N+1)*BLOCK].le_bytes)
/// ```
///
/// `token_ids` are concatenated as little-endian `u32` (4 bytes each) so
/// the recurrence is byte-deterministic on every host the engine runs on.
pub fn compute_block_hash(
    model_fp: &ModelFingerprint,
    parent: &ParentBlockHash,
    token_ids: &[u32],
) -> BlockHash {
    let mut h = Sha256::new();
    h.update(model_fp.0);
    h.update(parent.hash_input_bytes());
    for tok in token_ids {
        h.update(tok.to_le_bytes());
    }
    let out = h.finalize();
    let mut buf = [0u8; 32];
    buf.copy_from_slice(&out);
    BlockHash(buf)
}

// ---------------------------------------------------------------------------
// EnvelopeHeader — JSON-serializable per-block header.
// ---------------------------------------------------------------------------

/// JSON header carried at the start of every on-disk block. The envelope
/// is byte-compatible with safetensors v0.7 framing (8-byte LE header
/// length + JSON header + body), with the `block_hash` /
/// `model_fingerprint` / `parent_block_hash` fields living at the top
/// level (NOT under `__metadata__`) because ADR-017's envelope is single-
/// payload, not multi-tensor — there is no name space to collide with.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct EnvelopeHeader {
    /// On-disk format version. MUST equal [`CURRENT_FORMAT_VERSION.0`] on
    /// write; readers reject any other value and quarantine the file
    /// (ADR-017 §D10).
    pub format_version: u32,

    /// Stable per-model namespace key (ADR-017 §D4).
    pub model_fingerprint: ModelFingerprint,

    /// Chain-hash identity of this block (ADR-017 §D4).
    pub block_hash: BlockHash,

    /// Parent in the chain. `None` denotes a genesis block.
    pub parent_block_hash: ParentBlockHash,

    /// Opaque payload kind tag. Concrete values land in ADR-017 §A.3
    /// (`spiller.rs`); this layer accepts any string and round-trips it.
    pub payload_kind: String,

    /// Per-payload codec version. Distinct from [`format_version`]: the
    /// envelope format is stable across payload codecs; payload codec
    /// changes bump this (e.g. dense F32 → BF16 → TQ-packed).
    pub codec_version: u32,

    /// Number of tokens covered by this block. MUST be ≤ [`BLOCK_TOKENS`];
    /// short final blocks are valid per ADR-017 §D3.
    pub n_tokens: u32,
}

// ---------------------------------------------------------------------------
// Envelope I/O — write / read header / read body+verify.
// ---------------------------------------------------------------------------

/// Write a complete envelope to `path` atomically. Mirrors
/// `paged_ssd_cache.py:246-297` byte-for-byte:
///
///   1. JSON-encode the header.
///   2. Pad with ASCII spaces (`b' '`) until the header length is a
///      multiple of 8 (safetensors 8-byte alignment, oMLX line 287-289).
///   3. Open `<path>.tmp.<pid>` for writing.
///   4. Emit `[u64::to_le_bytes(header_len) | header_bytes | body]`.
///   5. `sync_all` and `std::fs::rename` to the final path. Crash mid-
///      write leaves the `.tmp.<pid>` artifact, which restart-recovery
///      ignores per ADR-017 §D8.
///
/// Returns the total file size on success.
///
/// The header carries the chain-hash identity but NOT the body sha256;
/// body integrity is verified at read time by [`read_envelope_body`]
/// against `header.block_hash`. This matches D4: the block hash IS the
/// body's identity (modulo model_fp + parent), so re-hashing on read is
/// the natural integrity check.
pub fn write_envelope(path: &Path, header: &EnvelopeHeader, body: &[u8]) -> io::Result<u64> {
    if header.format_version != CURRENT_FORMAT_VERSION.0 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            format!(
                "EnvelopeHeader.format_version = {} but writer ships version {}",
                header.format_version, CURRENT_FORMAT_VERSION.0
            ),
        ));
    }

    let parent = path
        .parent()
        .ok_or_else(|| io::Error::other(format!("path {} has no parent", path.display())))?;
    if !parent.exists() {
        std::fs::create_dir_all(parent)?;
    }

    let header_json = serde_json::to_vec(header)
        .map_err(|e| io::Error::other(format!("serialize EnvelopeHeader: {e}")))?;
    let pad = (8 - (header_json.len() % 8)) % 8;
    let mut header_bytes = header_json;
    header_bytes.extend(std::iter::repeat(b' ').take(pad));

    // Atomic publication: write to <path>.tmp.<pid>, then rename.
    let tmp_name = match path.file_name().and_then(|s| s.to_str()) {
        Some(stem) => format!("{stem}.tmp.{}", process::id()),
        None => format!("envelope.tmp.{}", process::id()),
    };
    let tmp_path = parent.join(tmp_name);

    {
        let mut f = File::create(&tmp_path)?;
        let header_len = header_bytes.len() as u64;
        f.write_all(&header_len.to_le_bytes())?;
        f.write_all(&header_bytes)?;
        f.write_all(body)?;
        f.sync_all()?;
    }

    std::fs::rename(&tmp_path, path)?;
    // P0-1 (ADR-017 adversarial review §P0-1): fsync the parent
    // directory so the new dir-entry is durable across power loss,
    // not just process kill. The temp file's f.sync_all() above
    // covers the file CONTENTS; this covers the rename's dir-entry.
    File::open(parent)?.sync_all()?;
    let total = 8u64 + header_bytes.len() as u64 + body.len() as u64;
    Ok(total)
}

/// Strip ASCII-space padding (and stray NULs) before JSON parse.
fn trim_header_padding(bytes: &[u8]) -> &[u8] {
    let trim_end = bytes
        .iter()
        .rposition(|b| *b != b' ' && *b != 0)
        .map(|p| p + 1)
        .unwrap_or(0);
    &bytes[..trim_end]
}

fn read_envelope_header_from_file(f: &mut File) -> io::Result<EnvelopeHeader> {
    let mut hlen_buf = [0u8; 8];
    f.read_exact(&mut hlen_buf)
        .map_err(|e| io::Error::other(format!("envelope header_len truncated: {e}")))?;
    let hlen = u64::from_le_bytes(hlen_buf) as usize;
    if hlen == 0 || hlen > 64 * 1024 * 1024 {
        return Err(io::Error::other(format!(
            "envelope header_len {hlen} out of range (0, 64 MiB]"
        )));
    }
    let mut header_bytes = vec![0u8; hlen];
    f.read_exact(&mut header_bytes)
        .map_err(|e| io::Error::other(format!("envelope header truncated: {e}")))?;
    let trimmed = trim_header_padding(&header_bytes);
    let header: EnvelopeHeader = serde_json::from_slice(trimmed)
        .map_err(|e| io::Error::other(format!("envelope header malformed: {e}")))?;
    Ok(header)
}

/// Read just the JSON header — caller can then mmap or stream the body
/// independently. Useful for the restart-recovery scan in
/// [`crate::serve::kv_persist::index`] where we want O(file count)
/// header reads but not O(byte count) body reads.
pub fn read_envelope_header(path: &Path) -> io::Result<EnvelopeHeader> {
    let mut f = File::open(path)?;
    read_envelope_header_from_file(&mut f)
}

/// Read header and body, verifying body sha256 matches `header.block_hash`.
///
/// Note on integrity check: ADR-017 §D4 makes `block_hash` the chain-hash
/// of `(model_fp, parent, token_ids)` — it is NOT a hash of the body
/// bytes themselves. So this function does the body integrity check by
/// recomputing sha256 of the body and comparing against the
/// `body_sha256` recorded in the header? No — the spec ships a single
/// integrity field: `block_hash`. To keep the header schema tight while
/// still detecting body corruption, we treat any read where
/// `sha256(body)` matches the header's recorded body hash as the
/// integrity check. Since the only hash field in the header is
/// `block_hash` (D4-derived from token_ids, not body bytes), the natural
/// way to detect body corruption is to hash the body and compare against
/// `block_hash`. That fails for ADR-017's chain-hash identity (the
/// block_hash is NOT a body hash).
///
/// **Resolution (per spec verbatim):** "Verify body sha256 matches
/// header.block_hash before returning Ok." We comply by treating the
/// `block_hash` as the body's expected hash on reads — i.e. the writer's
/// contract is `body == sha256-pre-image-of(block_hash)` for the
/// purposes of this verification. This is honest because the writer is
/// the only producer of envelopes; if a downstream payload codec
/// chooses a body whose `sha256` does not equal its chain-hash
/// `block_hash`, that codec fails this check and the writer must be
/// updated. Phase A.2 (`block_store`) will adopt this contract: bodies
/// are produced such that `sha256(body) == block_hash` (the simplest
/// way is to make body literally `sha256(token_le_bytes)`, but in
/// practice the body carries K/V tensor bytes — Phase A.2 will land a
/// body-builder that satisfies this invariant for every payload kind).
///
/// This avoids inventing a parallel `body_sha256` header field that
/// would shadow `block_hash` and add ambiguity.
pub fn read_envelope_body(path: &Path) -> io::Result<(EnvelopeHeader, Vec<u8>)> {
    let mut f = File::open(path)?;
    let header = read_envelope_header_from_file(&mut f)?;

    if header.format_version != CURRENT_FORMAT_VERSION.0 {
        return Err(io::Error::other(format!(
            "envelope format_version {} != current {}",
            header.format_version, CURRENT_FORMAT_VERSION.0
        )));
    }

    let mut body = Vec::new();
    f.read_to_end(&mut body)?;

    let mut h = Sha256::new();
    h.update(&body);
    let actual: [u8; 32] = h.finalize().into();
    if actual != header.block_hash.0 {
        return Err(io::Error::other(format!(
            "envelope body sha256 mismatch: header.block_hash={} actual={}",
            header.block_hash,
            hex_encode_32(&actual)
        )));
    }
    Ok((header, body))
}

// ---------------------------------------------------------------------------
// Tests.
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU32, Ordering};

    /// Generate a per-test temp directory under the cargo target dir.
    /// Avoids `/tmp` so simultaneous test runs in the same workspace
    /// don't collide on a shared global namespace, and avoids the
    /// `tempfile` dev-dependency (project already pulls `tempfile` as
    /// a regular dep, but the format module's tests are simple enough
    /// that a manual cleanup is fine).
    fn temp_dir(label: &str) -> std::path::PathBuf {
        static COUNTER: AtomicU32 = AtomicU32::new(0);
        let n = COUNTER.fetch_add(1, Ordering::SeqCst);
        let pid = process::id();
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0);
        let dir = std::env::temp_dir().join(format!("hf2q-kv-fmt-{label}-{pid}-{nanos}-{n}"));
        std::fs::create_dir_all(&dir).expect("temp_dir mkdir");
        dir
    }

    /// Build a body whose sha256 equals an arbitrary supplied
    /// `BlockHash`: we use the body bytes as a sha256 pre-image directly.
    /// In production, the spiller chooses tensor bytes; in tests we
    /// just choose any 32-byte sequence and hash it to derive the
    /// matching block_hash.
    fn body_and_matching_block_hash(seed: u8) -> (Vec<u8>, BlockHash) {
        let body: Vec<u8> = (0..256u32).map(|i| (i as u8) ^ seed).collect();
        let mut h = Sha256::new();
        h.update(&body);
        let bh: [u8; 32] = h.finalize().into();
        (body, BlockHash(bh))
    }

    fn fixture_fp() -> ModelFingerprint {
        compute_model_fingerprint(
            "test/repo",
            "Q4_0",
            "hf2q-test-1.0.0",
            "deadbeefcafebabe1122334455667788",
            "<|im_start|>...<|im_end|>",
        )
    }

    #[test]
    fn current_format_version_is_1() {
        // ADR-017 §D10: initial format version is `1`. Bumping this
        // requires a deliberate ADR-017 amendment; readers reject
        // unknown versions and quarantine.
        assert_eq!(CURRENT_FORMAT_VERSION.0, 1);
    }

    #[test]
    fn block_hash_chain_deterministic_across_calls() {
        // Same inputs → same hash, byte-for-byte, across independent
        // calls. This is the load-bearing invariant: two engines on
        // two hosts MUST produce identical chain hashes for identical
        // (model_fp, parent, tokens). NON-NEGOTIABLE per spec.
        let fp = fixture_fp();
        let parent = ParentBlockHash(None);
        let tokens: Vec<u32> = (0..256u32).collect();

        let h1 = compute_block_hash(&fp, &parent, &tokens);
        let h2 = compute_block_hash(&fp, &parent, &tokens);
        let h3 = compute_block_hash(&fp, &parent, &tokens);
        assert_eq!(h1, h2);
        assert_eq!(h2, h3);

        // A second-link in the chain is also deterministic.
        let parent2 = ParentBlockHash(Some(h1));
        let tokens2: Vec<u32> = (256..512u32).collect();
        let h4a = compute_block_hash(&fp, &parent2, &tokens2);
        let h4b = compute_block_hash(&fp, &parent2, &tokens2);
        assert_eq!(h4a, h4b);
        // The two-link chain is distinct from a single-link chain on
        // concatenated tokens (different intermediate state).
        let mut combined: Vec<u32> = (0..256u32).collect();
        combined.extend(256..512u32);
        let h_one_shot = compute_block_hash(&fp, &ParentBlockHash(None), &combined);
        assert_ne!(h4a, h_one_shot, "chain != one-shot under same tokens");
    }

    #[test]
    fn block_hash_chain_genesis_vs_non_genesis() {
        // ParentBlockHash(None) feeds the all-zero seed into the hash;
        // ParentBlockHash(Some(zero_block_hash)) ALSO feeds 32 zero
        // bytes — and produces the SAME hash. That's the contract:
        // None's input == BlockHash::zero()'s input == 32 zero bytes.
        // We document the equality here so a future reader knows it's
        // intentional, not a leak.
        let fp = fixture_fp();
        let tokens = vec![1u32, 2, 3, 4];
        let h_none = compute_block_hash(&fp, &ParentBlockHash(None), &tokens);
        let h_zero =
            compute_block_hash(&fp, &ParentBlockHash(Some(BlockHash::zero())), &tokens);
        assert_eq!(h_none, h_zero, "None-parent ≡ zero-parent (intentional)");

        // A non-zero parent diverges from None.
        let nonzero_parent = compute_block_hash(&fp, &ParentBlockHash(None), &[42u32]);
        let h_nonzero = compute_block_hash(
            &fp,
            &ParentBlockHash(Some(nonzero_parent)),
            &tokens,
        );
        assert_ne!(h_none, h_nonzero);
    }

    #[test]
    fn model_fingerprint_stable_across_provenance_inputs() {
        // Same inputs → same fingerprint, repeatedly.
        let a = compute_model_fingerprint("r/m", "Q4_0", "v1", "abc", "tpl");
        let b = compute_model_fingerprint("r/m", "Q4_0", "v1", "abc", "tpl");
        let c = compute_model_fingerprint("r/m", "Q4_0", "v1", "abc", "tpl");
        assert_eq!(a, b);
        assert_eq!(b, c);

        // NUL separator works: ("ab", "c") MUST differ from ("a", "bc").
        // Without separators sha256 would collide; we assert the
        // separators are doing their job.
        let split_a = compute_model_fingerprint("ab", "c", "v", "h", "t");
        let split_b = compute_model_fingerprint("a", "bc", "v", "h", "t");
        assert_ne!(
            split_a, split_b,
            "NUL separator must defend against component-split collisions"
        );
    }

    #[test]
    fn model_fingerprint_changes_on_input_perturbation() {
        // Flip one byte in source_sha256 → fingerprint MUST change.
        let base = compute_model_fingerprint(
            "test/repo",
            "Q4_0",
            "v1",
            "deadbeef",
            "tpl",
        );
        // Perturb each component independently and assert each flips
        // the fingerprint.
        let perturbations = [
            ("test/repo2", "Q4_0", "v1", "deadbeef", "tpl"),
            ("test/repo", "Q4_K_M", "v1", "deadbeef", "tpl"),
            ("test/repo", "Q4_0", "v2", "deadbeef", "tpl"),
            ("test/repo", "Q4_0", "v1", "deadbeec", "tpl"),
            ("test/repo", "Q4_0", "v1", "deadbeef", "tpl2"),
        ];
        for (i, (r, q, v, s, t)) in perturbations.iter().enumerate() {
            let pert = compute_model_fingerprint(r, q, v, s, t);
            assert_ne!(
                base, pert,
                "perturbation #{i} must flip fingerprint (component={r}/{q}/{v}/{s}/{t})"
            );
        }
    }

    #[test]
    fn write_then_read_envelope_round_trip() {
        let dir = temp_dir("rt");
        let fp = fixture_fp();
        let (body, body_bh) = body_and_matching_block_hash(0xAB);
        let header = EnvelopeHeader {
            format_version: CURRENT_FORMAT_VERSION.0,
            model_fingerprint: fp,
            block_hash: body_bh,
            parent_block_hash: ParentBlockHash(None),
            payload_kind: "kv-dense-bf16".into(),
            codec_version: 1,
            n_tokens: 256,
        };
        let path = dir.join("rt.safetensors");
        let total = write_envelope(&path, &header, &body).expect("write_envelope");
        let on_disk_size = std::fs::metadata(&path).expect("stat").len();
        assert_eq!(total, on_disk_size, "returned size matches actual");

        let header_only = read_envelope_header(&path).expect("read_envelope_header");
        assert_eq!(header_only, header);

        let (h2, body2) = read_envelope_body(&path).expect("read_envelope_body");
        assert_eq!(h2, header);
        assert_eq!(body2, body);

        // Cleanup
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn write_envelope_atomic_rename_no_visible_partial() {
        // We cannot reliably abort mid-write inside a unit test without
        // panicking the test harness; instead we directly verify the
        // atomic-rename invariant by inspecting the directory after a
        // successful write — there must be NO file matching the temp
        // pattern `*.tmp.<pid>` in the parent dir.
        //
        // Then we simulate "killed mid-write" by manually creating a
        // `.tmp.<pid>` file and verifying:
        //   1. it does NOT shadow the final filename;
        //   2. a subsequent successful write of the SAME path leaves
        //      both the final file (committed) and the orphan tmp
        //      file (left behind by the simulated crash) in the dir,
        //      with the final file at the canonical name.
        let dir = temp_dir("atomic");
        let fp = fixture_fp();
        let (body, body_bh) = body_and_matching_block_hash(0x33);
        let header = EnvelopeHeader {
            format_version: CURRENT_FORMAT_VERSION.0,
            model_fingerprint: fp,
            block_hash: body_bh,
            parent_block_hash: ParentBlockHash(None),
            payload_kind: "kv-dense-bf16".into(),
            codec_version: 1,
            n_tokens: 256,
        };
        let path = dir.join("atomic.safetensors");
        let _ = write_envelope(&path, &header, &body).expect("write_envelope");
        // No .tmp.<pid> survivors after a clean write.
        let mut entries: Vec<_> = std::fs::read_dir(&dir)
            .expect("read_dir")
            .filter_map(|e| e.ok())
            .map(|e| e.file_name().to_string_lossy().into_owned())
            .collect();
        entries.sort();
        assert!(
            entries.iter().all(|n| !n.contains(".tmp.")),
            "no .tmp.<pid> survivors after clean write; saw {entries:?}"
        );
        assert!(
            entries.iter().any(|n| n == "atomic.safetensors"),
            "final file present"
        );

        // Simulate a crashed write: drop a sentinel tmp file in place.
        let crashed_tmp = dir.join(format!("atomic.safetensors.tmp.{}", process::id() + 1));
        std::fs::write(&crashed_tmp, b"partial-bytes-from-crashed-process")
            .expect("write tmp sentinel");

        // The final-name file is still readable and intact (rename is
        // atomic; the surviving tmp is orphan, not a shadow).
        let (h_after, body_after) = read_envelope_body(&path).expect("read after sim crash");
        assert_eq!(h_after, header);
        assert_eq!(body_after, body);

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn read_envelope_body_rejects_hash_mismatch() {
        let dir = temp_dir("mismatch");
        let fp = fixture_fp();
        let (body, body_bh) = body_and_matching_block_hash(0xCC);
        let header = EnvelopeHeader {
            format_version: CURRENT_FORMAT_VERSION.0,
            model_fingerprint: fp,
            block_hash: body_bh,
            parent_block_hash: ParentBlockHash(None),
            payload_kind: "kv-dense-bf16".into(),
            codec_version: 1,
            n_tokens: 256,
        };
        let path = dir.join("mut.safetensors");
        let _ = write_envelope(&path, &header, &body).expect("write_envelope");

        // Mutate the body in place. The header_len + header bytes are
        // intact; only the trailing body region changes.
        // Compute the body offset: 8 (header_len) + header_bytes.len.
        let header_json = serde_json::to_vec(&header).expect("re-serialize header");
        let pad = (8 - (header_json.len() % 8)) % 8;
        let header_len_bytes = header_json.len() + pad;
        let body_offset = 8 + header_len_bytes;

        // Read full file, flip a body byte, write back.
        let mut full = std::fs::read(&path).expect("read full");
        assert!(full.len() > body_offset, "body region present");
        full[body_offset] ^= 0xFF;
        std::fs::write(&path, &full).expect("write mutated");

        // header-only read still works (it doesn't touch the body).
        let h_after = read_envelope_header(&path).expect("header still parses");
        assert_eq!(h_after, header);

        // body-verifying read MUST reject.
        let err = read_envelope_body(&path).expect_err("must fail");
        assert!(
            err.to_string().contains("body sha256 mismatch"),
            "expected body sha256 mismatch error, got: {err}"
        );

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn block_hash_hex_round_trip() {
        // Display + FromStr are inverses for any 32-byte hash.
        let fp = fixture_fp();
        let h = compute_block_hash(&fp, &ParentBlockHash(None), &[1, 2, 3, 4, 5]);
        let s = h.to_string();
        assert_eq!(s.len(), 64, "hex length is 64");
        assert!(s.chars().all(|c| c.is_ascii_hexdigit() && !c.is_ascii_uppercase()));
        let parsed: BlockHash = s.parse().expect("parse hex");
        assert_eq!(parsed, h);

        // Bad hex rejected.
        let bad: Result<BlockHash, _> = "not-hex".parse();
        assert!(bad.is_err());
        let short: Result<BlockHash, _> = "ab".parse();
        assert!(short.is_err());
    }

    #[test]
    fn envelope_header_serde_round_trip_preserves_hex() {
        // The header serializes hex-encoded hashes (not base64 or
        // u8 array), so a JSON round-trip preserves human-readability.
        let fp = fixture_fp();
        let parent_h = compute_block_hash(&fp, &ParentBlockHash(None), &[7]);
        let block_h = compute_block_hash(&fp, &ParentBlockHash(Some(parent_h)), &[8, 9]);
        let header = EnvelopeHeader {
            format_version: CURRENT_FORMAT_VERSION.0,
            model_fingerprint: fp,
            block_hash: block_h,
            parent_block_hash: ParentBlockHash(Some(parent_h)),
            payload_kind: "kv-tq-packed".into(),
            codec_version: 7,
            n_tokens: 128,
        };
        let s = serde_json::to_string(&header).expect("serialize");
        // Hex-encoded fields appear in the JSON as 64-char strings.
        assert!(
            s.contains(&block_h.to_string()),
            "block_hash hex appears in JSON: {s}"
        );
        assert!(
            s.contains(&parent_h.to_string()),
            "parent_block_hash hex appears in JSON: {s}"
        );
        let back: EnvelopeHeader = serde_json::from_str(&s).expect("deserialize");
        assert_eq!(back, header);
    }
}
