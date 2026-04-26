//! Tensor dump probe for the gemma4v ViT pipeline.
//!
//! ADR-005 Phase 2c iter 124 (W55) — Foundational parity probe.
//!
//! # Why
//!
//! Iter-122 fixed the (weight + 1) RMSNorm bug. Iter-123 (W54) audited four
//! more high-likelihood candidates against `/opt/llama.cpp/tools/mtmd/clip.cpp`
//! and the gemma4v graph builder; all matched byte-for-byte. The hypothesis
//! tree is exhausted and the smoke (`four_dots_in_corners_128x128.png`) still
//! diverges from llama-mtmd-cli's caption.
//!
//! W54's recommendation #5: stop guessing, start measuring. Build a
//! numerical parity probe that dumps tensors at every named ViT pipeline
//! stage so iter-125 can target the FIRST stage where hf2q diverges from a
//! peer reference.
//!
//! # Design
//!
//! 1. A thread-local `DumpCollector` holds optional `(name, MlxBuffer)`
//!    pairs accumulated during a forward pass.
//! 2. The forward function pushes ARC-clones of intermediate buffers into
//!    the collector when armed (shared-memory zero-copy on Apple Silicon).
//! 3. The CALLER arms the collector before invoking the forward, runs the
//!    pass, calls `session.finish()` (which waits on the GPU), then drains
//!    the collector and writes one `<stage>.bin` + `<stage>.json` per dump.
//! 4. Files are raw F32 little-endian, contiguous, with a JSON sidecar
//!    listing `shape`, `dtype`, `name`. No NPY: a 5-line raw dumper is
//!    adequate.
//!
//! # Trigger
//!
//! Set `HF2Q_VIT_DUMP=<dir>` to enable. Default unset → zero overhead, no
//! arc clones, no allocations. The check is a single `env::var` call per
//! forward via `LazyLock`; subsequent toggles are not honoured (matches the
//! existing `HF2Q_VIT_F32_ATTENTION` pattern).
//!
//! # Stages dumped
//!
//! ```text
//!  00_pre_patchify        — CPU input tensor [3, H, W] planar CHW
//!                            (matches peer's `inp_raw_scaled`; iter-126)
//!  00_post_patchify       — CPU input tensor [N_patches, P²·3] HWC
//!                            (hf2q-native; pre-iter-126 was named
//!                             `00_preprocess`)
//!  01_patch_embd          — patch_embed Linear output
//!  02_pos_embd            — post 2D pos_embd dual-table add
//!  03_block_NN            — output of ViT block NN, NN ∈ [0, num_layers)
//!  30_final_pool          — 3×3 spatial avg-pool output
//!  31_pool_sqrt_scale     — post sqrt(n_embd) scale
//!  32_std_bias_scale      — post (cur - std_bias) * std_scale
//!  33_projector           — Gemma4ClippableLinear (mm.0) output
//!  34_post_proj_rms       — final no-gain RMSNorm output (soft-token bus)
//! ```
//!
//! Stage indices use a 2-digit prefix so lexicographic ordering matches
//! pipeline order. Block stages use `03_block_00`..`03_block_NN` (3-digit
//! group is fine; outer prefix stays 03 for the entire transformer block
//! group).
//!
//! # Non-goals
//!
//! - Not a feature flag, not a `#[cfg]` switch — runtime env-gated only.
//!   Compiles into release builds with zero overhead when unset.
//! - Not a default behaviour. The forward path returns the same buffer
//!   regardless of dump state.

use std::cell::RefCell;
use std::path::{Path, PathBuf};
use std::sync::LazyLock;

use anyhow::{anyhow, Context, Result};
use mlx_native::MlxBuffer;

/// Process-wide one-shot env read for `HF2Q_VIT_DUMP`. Holds the resolved
/// dump directory when set, else `None`. Subsequent runtime mutations are
/// not honoured (matches the `HF2Q_VIT_F32_ATTENTION` pattern in
/// `vit_gpu.rs`).
///
/// We re-read on each `dump_dir()` call rather than caching, because the
/// caller (test, end-to-end CLI) may set the env at startup and we want
/// child threads in the same process to see it. `std::env::var` is cheap
/// (microseconds) and only fires when the collector is armed.
fn dump_dir_env() -> Option<PathBuf> {
    match std::env::var("HF2Q_VIT_DUMP") {
        Ok(s) if !s.is_empty() => Some(PathBuf::from(s)),
        _ => None,
    }
}

// ADR-005 Phase 2c iter 130 (W61) — comprehensive dtype audit instrumentation.
//
// `HF2Q_VIT_DUMP_DTYPE_AUDIT=1` is a development-only env var that, when
// armed alongside `HF2Q_VIT_DUMP=<dir>`, instructs the recording sites in
// `gemma4v_block_forward_gpu` and the attention helpers to also append a
// `(name, dtype, shape)` triple into a thread-local audit collector. After
// the forward pass completes, the caller drains the audit collector and
// writes a single `_dtype_audit.json` file listing every observed
// intermediate. INSTRUMENTATION ONLY — no production behavior change.
//
// Why this is its own env var separate from `HF2Q_VIT_DUMP`: the audit
// path hits buffers we don't otherwise dump as F32 binaries (e.g. the
// internal F16 staging tiles `k_f16`, `v_f16`, `v_t_f16` inside
// `vit_attention_gpu`). Those buffers can't be written as F32 raw binaries
// without additional logic; the audit just reads `buffer.dtype()` and
// `buffer.shape()` (both cheap accessors with no kernel work).
//
// Read once at first access (LazyLock); subsequent runtime mutations
// not honoured. Matches the `HF2Q_VIT_F32_ATTENTION` and `HF2Q_VIT_DUMP`
// patterns in this module.
fn dtype_audit_env() -> bool {
    matches!(std::env::var("HF2Q_VIT_DUMP_DTYPE_AUDIT").as_deref(), Ok("1"))
}

thread_local! {
    /// When `Some`, the active forward pass appends `(stage_name, MlxBuffer)`
    /// here for the caller to drain and write after `session.finish()`.
    /// When `None`, dump points are no-ops.
    static COLLECTOR: RefCell<Option<Vec<(String, MlxBuffer)>>> = const {
        RefCell::new(None)
    };

    /// ADR-005 iter 130 (W61) — dtype audit collector (separate from
    /// `COLLECTOR`). Holds `(name, dtype_str, shape)` triples appended by
    /// `record_audit` when both `HF2Q_VIT_DUMP=<dir>` AND
    /// `HF2Q_VIT_DUMP_DTYPE_AUDIT=1` are set. Drained by
    /// `drain_audit_entries` after `with_dump_collector` exits.
    static AUDIT_COLLECTOR: RefCell<Vec<AuditEntry>> = const {
        RefCell::new(Vec::new())
    };
}

/// One entry of the dtype audit. Recorded per intermediate MlxBuffer in
/// the gemma4v block forward path when audit instrumentation is armed.
#[derive(Debug, Clone)]
pub struct AuditEntry {
    pub name: String,
    pub dtype: String,
    pub shape: Vec<usize>,
}

/// Arm the thread-local collector for the duration of the closure, then
/// drain and return the collected `(name, buffer)` pairs.
///
/// The forward pass should be invoked inside the closure. After the
/// closure returns, the collector is unset and the caller can write the
/// dumps.
///
/// # Errors
///
/// Propagates errors from the closure. Returns an error if the collector
/// is already armed (re-entrant calls are not supported).
pub fn with_dump_collector<F, R>(f: F) -> Result<(R, Vec<(String, MlxBuffer)>)>
where
    F: FnOnce() -> Result<R>,
{
    COLLECTOR.with(|c| {
        if c.borrow().is_some() {
            return Err(anyhow!(
                "with_dump_collector: re-entrant arming not supported"
            ));
        }
        *c.borrow_mut() = Some(Vec::new());
        Ok(())
    })?;
    let result = f();
    let collected = COLLECTOR.with(|c| c.borrow_mut().take().unwrap_or_default());
    let r = result?;
    Ok((r, collected))
}

/// Append `(name, buffer)` to the collector if armed. No-op otherwise.
///
/// Callers inside `gemma4v_apply_full_forward_gpu` invoke this at each
/// named stage. The buffer is ARC-cloned (Apple Silicon shared-memory:
/// `MlxBuffer::clone` increments the Metal retain count, no data copy).
///
/// Stage names should follow the canonical taxonomy in this module's
/// header doc — lexicographic prefix encodes pipeline order.
pub fn record(name: &str, buffer: &MlxBuffer) {
    COLLECTOR.with(|c| {
        if let Some(v) = c.borrow_mut().as_mut() {
            v.push((name.to_string(), buffer.clone()));
        }
    });
}

/// Returns true if the dump collector is armed for the current thread.
///
/// Callers use this to gate optional pre-stage CPU mirrors (e.g. dumping
/// the pre-GPU `preprocess_gemma4v` output, which never lives on the
/// device). When false, the entire dump path is bypassed and incurs no
/// overhead beyond the `RefCell::borrow`.
pub fn is_armed() -> bool {
    COLLECTOR.with(|c| c.borrow().is_some())
}

/// Returns true if the dtype-audit instrumentation is enabled.
///
/// ADR-005 iter 130 (W61). One-shot env read of
/// `HF2Q_VIT_DUMP_DTYPE_AUDIT=1` cached in a `LazyLock`; subsequent
/// runtime mutations are NOT honoured (matches the
/// `HF2Q_VIT_F32_ATTENTION` pattern in vit_gpu.rs).
///
/// Callers in `vit_gpu.rs::gemma4v_block_forward_gpu` /
/// `vit_attention_gpu` / `vit_attention_scores_gpu` gate `record_audit`
/// calls behind both this AND `is_armed()` so the audit only fires
/// inside an `with_dump_collector` scope (the audit's drain path runs
/// alongside the dump-buffer drain at scope exit).
pub fn is_dtype_audit_armed() -> bool {
    static AUDIT_ARMED: LazyLock<bool> = LazyLock::new(dtype_audit_env);
    *AUDIT_ARMED
}

/// Append a `(name, dtype, shape)` triple to the dtype-audit collector.
///
/// No-op when `is_dtype_audit_armed()` is false OR when `is_armed()`
/// (the GPU-buffer collector) is false. The audit collector is drained
/// alongside the dump-buffer collector at `with_dump_collector` scope
/// exit and written via `write_dtype_audit`.
///
/// # Cost
///
/// `buffer.dtype()` + `buffer.shape()` are constant-time accessors; the
/// `Vec::push` is amortized O(1). Total cost per call: ~1 microsecond
/// (string format dominates). Scaled across 27 blocks × ~33
/// intermediates per block = ~890 calls per forward = <1 ms total.
pub fn record_audit(name: &str, buffer: &MlxBuffer) {
    if !is_dtype_audit_armed() {
        return;
    }
    if !is_armed() {
        return;
    }
    let dtype = format!("{:?}", buffer.dtype());
    let shape = buffer.shape().to_vec();
    AUDIT_COLLECTOR.with(|c| {
        c.borrow_mut().push(AuditEntry {
            name: name.to_string(),
            dtype,
            shape,
        });
    });
}

/// Drain the dtype-audit collector and return all accumulated entries.
/// Caller (parity probe runner) writes the JSON via `write_dtype_audit`
/// after `with_dump_collector` returns.
pub fn drain_audit_entries() -> Vec<AuditEntry> {
    AUDIT_COLLECTOR.with(|c| std::mem::take(&mut *c.borrow_mut()))
}

/// Write the dtype-audit JSON sidecar to `<dir>/_dtype_audit.json`.
///
/// Format (one JSON array of `{name, dtype, shape}` objects):
///
/// ```text
/// [
///   {"name":"03_block_00_01_pre_attn_norm","dtype":"F32","shape":[196,1152]},
///   {"name":"03_block_00_attn_q_proj","dtype":"F32","shape":[196,1152]},
///   ...
/// ]
/// ```
///
/// Hand-rolled JSON to avoid a new dependency (matches `write_dump_inner`
/// pattern). One file per forward pass, overwritten if it already exists.
///
/// # Errors
///
/// I/O errors creating or writing the file.
pub fn write_dtype_audit(dir: &Path, entries: &[AuditEntry]) -> Result<()> {
    use std::io::Write;
    let path = dir.join("_dtype_audit.json");
    let mut f = std::fs::File::create(&path)
        .with_context(|| format!("create {}", path.display()))?;
    f.write_all(b"[\n")
        .with_context(|| format!("write {}", path.display()))?;
    for (i, e) in entries.iter().enumerate() {
        let shape_str = e
            .shape
            .iter()
            .map(|n| n.to_string())
            .collect::<Vec<_>>()
            .join(",");
        let line = format!(
            "  {{\"name\":\"{}\",\"dtype\":\"{}\",\"shape\":[{}]}}",
            e.name, e.dtype, shape_str
        );
        f.write_all(line.as_bytes())
            .with_context(|| format!("write entry to {}", path.display()))?;
        if i + 1 < entries.len() {
            f.write_all(b",\n")
                .with_context(|| format!("write {}", path.display()))?;
        } else {
            f.write_all(b"\n")
                .with_context(|| format!("write {}", path.display()))?;
        }
    }
    f.write_all(b"]\n")
        .with_context(|| format!("write {}", path.display()))?;
    Ok(())
}


/// Record a CPU-side `&[f32]` slice as a synthetic stage. Used for the
/// `00_pre_patchify` / `00_post_patchify` dumps (input tensors that are
/// CPU `Vec<f32>` derived from `preprocess_gemma4v`, never live on the
/// device).
///
/// We allocate a one-off f32 byte buffer in a side-store and stash it
/// alongside the GPU buffers. Same on-disk format as GPU dumps.
pub fn record_f32(name: &str, data: &[f32], shape: Vec<usize>) {
    COLLECTOR.with(|c| {
        if let Some(_v) = c.borrow_mut().as_mut() {
            // Stash via the side-channel CPU collector: we write
            // immediately to a parallel pending list keyed by name. To
            // keep the data structure uniform with the GPU collector, we
            // route through a thread-local CPU-mirror map drained by
            // `drain_cpu_mirrors`.
            CPU_MIRRORS.with(|m| {
                m.borrow_mut().push(CpuMirror {
                    name: name.to_string(),
                    data: data.to_vec(),
                    shape,
                });
            });
        }
    });
}

/// One CPU-only stage dump. Held in a parallel thread-local because
/// `MlxBuffer` only wraps GPU buffers; we don't want to round-trip CPU
/// data through a Metal allocation just to satisfy a uniform container.
pub struct CpuMirror {
    pub name: String,
    pub data: Vec<f32>,
    pub shape: Vec<usize>,
}

thread_local! {
    static CPU_MIRRORS: RefCell<Vec<CpuMirror>> = const { RefCell::new(Vec::new()) };
}

/// Drain the CPU-mirror collector. Called after `with_dump_collector`
/// returns, alongside the GPU-buffer collection. Empties the side store.
pub fn drain_cpu_mirrors() -> Vec<CpuMirror> {
    CPU_MIRRORS.with(|m| std::mem::take(&mut *m.borrow_mut()))
}

/// Resolve the dump directory from the env (creating it if needed) and
/// return it. `None` when `HF2Q_VIT_DUMP` is unset.
pub fn resolve_dump_dir() -> Result<Option<PathBuf>> {
    let Some(dir) = dump_dir_env() else {
        return Ok(None);
    };
    if !dir.exists() {
        std::fs::create_dir_all(&dir)
            .with_context(|| format!("create dump dir {}", dir.display()))?;
    } else if !dir.is_dir() {
        return Err(anyhow!(
            "HF2Q_VIT_DUMP={} exists but is not a directory",
            dir.display()
        ));
    }
    Ok(Some(dir))
}

/// Write a single dump as `<dir>/<name>.bin` (raw F32 LE) plus
/// `<dir>/<name>.json` (sidecar shape metadata). Only F32 buffers are
/// supported — every gemma4v ViT activation is F32 (the BF16 K cast
/// inside attention is internal to the attention kernel and does not
/// surface as an inter-stage tensor).
///
/// # Errors
///
/// I/O errors creating files; dtype mismatch (non-F32 buffer).
pub fn write_dump_gpu(dir: &Path, name: &str, buffer: &MlxBuffer) -> Result<()> {
    use mlx_native::DType;
    if buffer.dtype() != DType::F32 {
        return Err(anyhow!(
            "write_dump_gpu({name}): expected F32, got {:?}",
            buffer.dtype()
        ));
    }
    // SAFETY: we just confirmed dtype = F32; buffer is in shared memory
    // on Apple Silicon and the GPU work that produced it has been
    // awaited (caller invokes us only after `session.finish()`).
    let slice: &[f32] = buffer
        .as_slice::<f32>()
        .map_err(|e| anyhow!("write_dump_gpu({name}): as_slice: {e}"))?;
    write_dump_inner(dir, name, slice, buffer.shape())
}

/// Write a CPU mirror `Vec<f32>` to disk. Same on-disk format as
/// `write_dump_gpu`.
pub fn write_dump_cpu(dir: &Path, mirror: &CpuMirror) -> Result<()> {
    write_dump_inner(dir, &mirror.name, &mirror.data, &mirror.shape)
}

fn write_dump_inner(dir: &Path, name: &str, data: &[f32], shape: &[usize]) -> Result<()> {
    let bin_path = dir.join(format!("{name}.bin"));
    let json_path = dir.join(format!("{name}.json"));

    // Raw F32 LE bytes. bytemuck would be cleaner but adds a dep nudge;
    // explicit little-endian on x86 / aarch64 is the same as
    // memcpy-of-f32-slice. Stay portable: use to_le_bytes().
    use std::io::Write;
    let mut file = std::fs::File::create(&bin_path)
        .with_context(|| format!("create {}", bin_path.display()))?;
    let mut buf = Vec::with_capacity(data.len() * 4);
    for v in data {
        buf.extend_from_slice(&v.to_le_bytes());
    }
    file.write_all(&buf)
        .with_context(|| format!("write {}", bin_path.display()))?;

    // JSON sidecar — minimal, hand-rolled to avoid a new dep.
    let shape_str = shape
        .iter()
        .map(|n| n.to_string())
        .collect::<Vec<_>>()
        .join(",");
    let json = format!(
        "{{\"name\":\"{}\",\"dtype\":\"f32\",\"shape\":[{}],\"n_elements\":{}}}\n",
        name,
        shape_str,
        data.len()
    );
    let mut jf = std::fs::File::create(&json_path)
        .with_context(|| format!("create {}", json_path.display()))?;
    jf.write_all(json.as_bytes())
        .with_context(|| format!("write {}", json_path.display()))?;
    Ok(())
}

/// Stable check the env-var pattern used by other vit_gpu env-gates.
/// Provided here so callers don't drift in env-var spelling.
#[allow(dead_code)]
pub fn env_var_name() -> &'static str {
    "HF2Q_VIT_DUMP"
}

/// One-shot `LazyLock`-backed snapshot of `HF2Q_VIT_DUMP` taken at first
/// access. Mirrors the `HF2Q_VIT_F32_ATTENTION` style; intended for
/// tests/diagnostics that want a single resolved path without re-reading
/// env on every call. Currently unused — public for future ergonomic
/// callers.
#[allow(dead_code)]
pub static DUMP_DIR_ONESHOT: LazyLock<Option<PathBuf>> = LazyLock::new(dump_dir_env);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn collector_unarmed_no_op() {
        // Without with_dump_collector, record() must be a silent no-op.
        // We can't easily synthesize an MlxBuffer here without a Metal
        // device, so just exercise the CPU path.
        record_f32("00_test_unarmed", &[1.0, 2.0, 3.0], vec![3]);
        // CPU mirror must be empty since collector wasn't armed.
        let drained = drain_cpu_mirrors();
        assert!(
            drained.is_empty(),
            "CPU mirrors should be empty when collector is unarmed"
        );
    }

    #[test]
    fn collector_armed_collects_cpu_mirror() {
        let result: Result<()> = with_dump_collector(|| {
            record_f32("00_test_armed", &[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
            Ok(())
        })
        .map(|((), _)| ());
        result.expect("armed collect");
        let mirrors = drain_cpu_mirrors();
        assert_eq!(mirrors.len(), 1);
        assert_eq!(mirrors[0].name, "00_test_armed");
        assert_eq!(mirrors[0].shape, vec![2, 2]);
        assert_eq!(mirrors[0].data, vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn write_dump_inner_round_trip() {
        let tmp = std::env::temp_dir().join(format!(
            "vit_dump_test_{}",
            std::process::id()
        ));
        std::fs::create_dir_all(&tmp).expect("mkdir tmp");
        let data = vec![1.5_f32, -2.25, 3.75, 0.0];
        let shape = vec![2, 2];
        write_dump_inner(&tmp, "01_round_trip", &data, &shape).expect("write");
        let bin = std::fs::read(tmp.join("01_round_trip.bin")).expect("read bin");
        assert_eq!(bin.len(), data.len() * 4);
        // Spot-check first f32 LE.
        let f0 = f32::from_le_bytes([bin[0], bin[1], bin[2], bin[3]]);
        assert_eq!(f0, 1.5);
        let json = std::fs::read_to_string(tmp.join("01_round_trip.json")).expect("read json");
        assert!(json.contains("\"shape\":[2,2]"));
        assert!(json.contains("\"dtype\":\"f32\""));
        assert!(json.contains("\"n_elements\":4"));
        std::fs::remove_dir_all(&tmp).ok();
    }
}
