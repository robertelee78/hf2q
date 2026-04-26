//! Helpers for ADR-009/010 investigation-mode dumps.
//!
//! The decode/prefill hot paths historically grew ~20-line `if
//! dump_layers { s.finish(); let data = buf.as_slice()?; let dump_dir
//! = ...; let path = format!(...); let bytes = unsafe { slice_from_raw
//! }; fs::write; eprintln; s = exec.begin() }` blocks — one per
//! diagnostic buffer. The fs::write + slice-from-raw-ptr part is
//! mechanical and identical across sites; this module centralizes it.
//!
//! The *session dance* (finish → dump → re-begin) stays inline in the
//! call sites because it needs `s` and `exec` which are awkward to
//! pass through, and because batching multiple dumps between one
//! finish/begin pair is an important perf optimization that the
//! caller controls.
//!
//! Files are written to [`INVESTIGATION_ENV.dump_dir`]
//! (`HF2Q_DUMP_DIR`, default `/tmp`).

#![allow(dead_code)]

use anyhow::Result;
use mlx_native::MlxBuffer;

use super::INVESTIGATION_ENV;

/// Write `n_bytes` raw U8 values from `buf` to
/// `<dump_dir>/hf2q_<name>_layer<LL>_pos<seq_pos>.u8.bin` and emit one
/// `[DUMP]` line on stderr. Used for TQ packed K/V cache (nibble-packed
/// u8 arrays). The caller is responsible for finishing the GPU session
/// before calling so the buffer contents are valid.
pub fn dump_u8(
    buf: &MlxBuffer,
    n_bytes: usize,
    name: &str,
    layer_idx: usize,
    seq_pos: usize,
) -> Result<()> {
    let data: &[u8] = buf
        .as_slice()
        .map_err(|e| anyhow::anyhow!("dump_u8 {name} read: {e}"))?;
    let dump_dir = &INVESTIGATION_ENV.dump_dir;
    let path = format!("{dump_dir}/hf2q_{name}_layer{layer_idx:02}_pos{seq_pos}.u8.bin");
    std::fs::write(&path, &data[..n_bytes])
        .map_err(|e| anyhow::anyhow!("write {path}: {e}"))?;
    eprintln!("[DUMP] {name} layer {layer_idx:02} ({n_bytes} u8) -> {path}");
    Ok(())
}

/// Write a meta JSON sidecar to
/// `<dump_dir>/hf2q_<name>_layer<LL>_pos<seq_pos>.json`.
/// `json_str` should be a pretty-printed JSON string (use `serde_json::to_string_pretty`).
pub fn dump_meta_json(
    json_str: &str,
    name: &str,
    layer_idx: usize,
    seq_pos: usize,
) -> Result<()> {
    let dump_dir = &INVESTIGATION_ENV.dump_dir;
    let path = format!("{dump_dir}/hf2q_{name}_layer{layer_idx:02}_pos{seq_pos}.json");
    std::fs::write(&path, json_str.as_bytes())
        .map_err(|e| anyhow::anyhow!("write {path}: {e}"))?;
    eprintln!("[DUMP] {name} layer {layer_idx:02} -> {path}");
    Ok(())
}

/// Write `n_elems` F32 values from `buf` to
/// `<dump_dir>/hf2q_<name>[_layer<NN>]_pos<seq_pos>.bin` and emit one
/// `[DUMP]` line on stderr. `layer_idx = None` produces the
/// layer-less form used for pre-lm_head / post-lm_head dumps.
///
/// Invariant: `buf` must be a host-readable F32 buffer containing at
/// least `n_elems` elements. The caller is responsible for finishing
/// the GPU session before calling so the buffer contents are valid.
pub fn dump_f32(
    buf: &MlxBuffer,
    n_elems: usize,
    name: &str,
    layer_idx: Option<usize>,
    seq_pos: usize,
) -> Result<()> {
    dump_f32_to(buf, n_elems, name, layer_idx, seq_pos, None)
}

/// W39 iter-112b: explicit-directory variant of [`dump_f32`].  `dir_override =
/// Some(_)` writes the file under that path instead of `INVESTIGATION_ENV
/// .dump_dir`; `None` falls back to the LazyLock value (i.e. byte-identical
/// to [`dump_f32`]).
///
/// Motivation: `INVESTIGATION_ENV` is populated by `LazyLock` at process
/// start (`main.rs::main`) before `Cli::parse`, so a runtime `set_var`
/// after `INVESTIGATION_ENV.activate()` is too late to reach the LazyLock.
/// The Gate H two-regime in-process harness needs to redirect dumps at
/// run time without restarting the process; this overload is the
/// per-call escape hatch.
pub fn dump_f32_to(
    buf: &MlxBuffer,
    n_elems: usize,
    name: &str,
    layer_idx: Option<usize>,
    seq_pos: usize,
    dir_override: Option<&std::path::Path>,
) -> Result<()> {
    let data: &[f32] = buf
        .as_slice()
        .map_err(|e| anyhow::anyhow!("dump {name} read: {e}"))?;
    let dump_dir_owned: String;
    let dump_dir: &str = match dir_override {
        Some(p) => {
            dump_dir_owned = p.to_string_lossy().into_owned();
            &dump_dir_owned
        }
        None => &INVESTIGATION_ENV.dump_dir,
    };
    let path = match layer_idx {
        Some(l) => format!("{dump_dir}/hf2q_{name}_layer{l:02}_pos{seq_pos}.bin"),
        None => format!("{dump_dir}/hf2q_{name}_pos{seq_pos}.bin"),
    };
    let bytes: &[u8] = unsafe {
        std::slice::from_raw_parts(
            data.as_ptr() as *const u8,
            n_elems * std::mem::size_of::<f32>(),
        )
    };
    std::fs::write(&path, bytes)
        .map_err(|e| anyhow::anyhow!("write {path}: {e}"))?;
    match layer_idx {
        Some(l) => eprintln!("[DUMP] {name} layer {l:02} ({n_elems} f32) -> {path}"),
        None => eprintln!("[DUMP] {name} ({n_elems} f32) -> {path}"),
    }
    Ok(())
}
