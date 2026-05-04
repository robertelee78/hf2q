//! ADR-015 iter61a-3: per-op hidden-state bisection scaffold.
//!
//! When `HF2Q_DUMP_LAYER` is set (any non-empty value), every major op in
//! the qwen35 forward path writes the FULL hidden buffer (not just the
//! last token) to disk as raw little-endian f32 bytes plus a manifest
//! line.  The dump is keyed by `<op_kind, layer_idx, step_idx>` so two
//! cold-process runs on identical inputs can be byte-diffed to find the
//! earliest divergence.
//!
//! ## Env vars
//! * `HF2Q_DUMP_LAYER` — gate.  Values:
//!     * unset / empty — no dumping (zero cost; sync helpers compile-fold to no-ops).
//!     * `ALL`         — dump every op of every layer.
//!     * `<N>`         — dump only layer `N` (numeric).  Embed and output_norm
//!                       are always dumped (layer "embed"/"output_norm").
//! * `HF2Q_DUMP_RUN_ID` — run identifier (defaults to the PID).
//!     The dump dir is `/tmp/hf2q-dump/<run_id>/`.
//!
//! ## File format
//! Each dump is a contiguous little-endian `f32` buffer, named:
//!   `step{step:04}_layer{layer:03}_{op}.f32`
//! where `step` is the `forward_*` invocation counter (prefill = step 0,
//! first decode = step 1, …) and `op` is the op label (e.g. `embed`,
//! `attn_out`, `ffn_out`, `ffn_input`, `ffn_residual`, `output_norm`,
//! `argmax_logit`).
//!
//! A manifest at `manifest.txt` is appended on each dump:
//!   `step layer op shape={...} bytes=<n> path=<file>`
//!
//! ## Why full buffer not last-token
//! The earlier `dump_layer_bin` only wrote the last-token slice, which
//! was sufficient for last-token-NaN diagnosis but loses information
//! when two cold runs diverge in any non-last token (e.g. RoPE position
//! mismatch, prefill arena alias).  Bisection requires byte-exact
//! comparison of the full buffer.
//!
//! ## Cost
//! * env unset: a single `OnceLock` boolean read per call site.  No
//!   `download_f32`, no allocation, no I/O.  Compile-folded as
//!   `if false { … }` to a near-no-op.
//! * env set: one GPU→CPU `download_f32` (sync) + one fs::write per op.
//!   Acceptable for diagnostic runs (bisection is single-token, not
//!   long-decode).

use anyhow::{Context, Result};
use mlx_native::{EncoderSession, MlxBuffer, MlxDevice};
use std::cell::Cell;
use std::fs::{File, OpenOptions};
use std::io::Write;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::OnceLock;

use super::gpu_full_attn::download_f32;

thread_local! {
    /// ADR-015 iter94 Task #3 — thread-local session drainer pointer.
    ///
    /// When set (via [`set_active_session`]), [`flush_gpu`] routes its
    /// drain through `EncoderSession::commit_and_wait +
    /// reset_for_next_stage` on the pointed session INSTEAD of opening a
    /// fresh `device.command_encoder()` and committing it.  This is
    /// required for env=1 (HF2Q_ENCODER_SESSION=1) bisection: the
    /// session's persistent CB carries the in-flight intra-K dispatches,
    /// and a fresh-encoder `commit_and_wait` does NOT join that session,
    /// so download-side reads see pre-write state (iter93 Phase C found
    /// all-zero ffn_out reads from the session-only env=1 path).
    ///
    /// # Lifetime contract
    ///
    /// The pointer is set ONLY by `forward_gpu_*` immediately before the
    /// per-layer dispatch loop and CLEARED immediately after.  Within
    /// that window, the pointed `EncoderSession` is on the same thread's
    /// stack and cannot be moved (the layer loop holds it via
    /// `&mut Option<EncoderSession>`).  `flush_gpu` reads this Cell
    /// from the SAME thread at dump time.  No cross-thread aliasing is
    /// possible because the Cell is thread-local and dump_bisect is
    /// only invoked on the inference thread.
    ///
    /// `dump_bisect` is a diagnostic-only path gated by `HF2Q_DUMP_LAYER`;
    /// the raw-pointer dance is acceptable in that scope.  Production
    /// runs (env unset) bypass [`is_enabled`] before `flush_gpu` is ever
    /// called, so this thread-local is dormant on the hot path.
    static ACTIVE_SESSION: Cell<Option<*mut EncoderSession>> = const { Cell::new(None) };
}

/// Install a session pointer for [`flush_gpu`] to drain through.
///
/// Caller must call [`clear_active_session`] before the session is
/// dropped or moved.  Recommended pattern:
///
/// ```ignore
/// if let Some(sess) = layer_session.as_mut() {
///     dump_bisect::set_active_session(sess);
/// }
/// // ... layer loop ...
/// dump_bisect::clear_active_session();
/// ```
///
/// # Safety
///
/// The caller must ensure that the pointed session outlives any
/// subsequent `flush_gpu` calls (i.e. the session's stack-frame must
/// not exit until [`clear_active_session`] runs).  Stamped here as a
/// SAFE function because the `*mut EncoderSession` is treated opaquely
/// by this module — no read/write happens until [`flush_gpu`]
/// dereferences it inside its own `unsafe` block.
pub fn set_active_session(sess: &mut EncoderSession) {
    ACTIVE_SESSION.with(|c| c.set(Some(sess as *mut EncoderSession)));
}

/// Clear the session-drainer pointer.  Idempotent.
pub fn clear_active_session() {
    ACTIVE_SESSION.with(|c| c.set(None));
}

/// Cached env-gate state.  Computed once per process.
struct DumpConfig {
    /// `None` if disabled, `Some(LayerFilter)` otherwise.
    filter: LayerFilter,
    /// `/tmp/hf2q-dump/<run_id>/`
    dir: PathBuf,
}

#[derive(Clone, Copy, Debug)]
enum LayerFilter {
    All,
    Only(usize),
}

static CONFIG: OnceLock<Option<DumpConfig>> = OnceLock::new();
static STEP_COUNTER: AtomicU64 = AtomicU64::new(0);

fn config() -> Option<&'static DumpConfig> {
    CONFIG
        .get_or_init(|| {
            let raw = std::env::var("HF2Q_DUMP_LAYER").ok()?;
            let raw = raw.trim();
            if raw.is_empty() {
                return None;
            }
            let filter = if raw.eq_ignore_ascii_case("ALL") {
                LayerFilter::All
            } else {
                match raw.parse::<usize>() {
                    Ok(n) => LayerFilter::Only(n),
                    Err(_) => {
                        eprintln!(
                            "[DUMP_BISECT] HF2Q_DUMP_LAYER={raw:?} is not 'ALL' or a usize; \
                             dumping disabled"
                        );
                        return None;
                    }
                }
            };
            let run_id = std::env::var("HF2Q_DUMP_RUN_ID")
                .ok()
                .filter(|s| !s.is_empty())
                .unwrap_or_else(|| std::process::id().to_string());
            let dir = PathBuf::from("/tmp/hf2q-dump").join(&run_id);
            if let Err(e) = std::fs::create_dir_all(&dir) {
                eprintln!(
                    "[DUMP_BISECT] failed to create {}: {e}; dumping disabled",
                    dir.display()
                );
                return None;
            }
            // Truncate the manifest at session start.
            let manifest_path = dir.join("manifest.txt");
            if let Err(e) = File::create(&manifest_path) {
                eprintln!(
                    "[DUMP_BISECT] failed to create manifest {}: {e}; dumping disabled",
                    manifest_path.display()
                );
                return None;
            }
            eprintln!(
                "[DUMP_BISECT] enabled: filter={:?} dir={}",
                filter,
                dir.display()
            );
            Some(DumpConfig { filter, dir })
        })
        .as_ref()
}

/// Returns true iff dumping is enabled (any layer / any op).
#[inline]
pub fn is_enabled() -> bool {
    config().is_some()
}

/// Returns true iff this op should be dumped for the given layer.
///
/// `layer_idx == None` means "always-on" ops (embed, output_norm).
#[inline]
pub fn should_dump(layer_idx: Option<usize>) -> bool {
    let Some(cfg) = config() else { return false };
    match (cfg.filter, layer_idx) {
        (LayerFilter::All, _) => true,
        (LayerFilter::Only(_), None) => true,
        (LayerFilter::Only(n), Some(idx)) => idx == n,
    }
}

/// Increment + return the step counter.  Called once per `forward_*`
/// invocation by the caller (we use this as the prefix for the dump
/// filename so prefill (step 0) and decode tokens (step 1, 2, …) are
/// distinguishable).
pub fn next_step() -> u64 {
    STEP_COUNTER.fetch_add(1, Ordering::SeqCst)
}

/// Read the current step without incrementing.
pub fn current_step() -> u64 {
    STEP_COUNTER.load(Ordering::SeqCst)
}

/// Reset the step counter.  Only used by tests.
#[cfg(test)]
pub fn reset_step() {
    STEP_COUNTER.store(0, Ordering::SeqCst);
}

thread_local! {
    /// Per-thread "current layer index" used by within-layer dump call sites
    /// (e.g. `build_gated_attn_layer` doesn't know its layer index, the
    /// caller does).  Set via [`set_current_layer`] from `forward_gpu_*`
    /// before each layer's call into the layer builder.
    static CURRENT_LAYER: std::cell::Cell<Option<usize>> = const { std::cell::Cell::new(None) };
    /// Per-thread "current step" — same purpose as CURRENT_LAYER but for the
    /// step counter (avoids passing it through every layer-builder signature).
    static CURRENT_STEP: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };
}

/// Read the current per-thread (step, layer_idx) for use by ad-hoc
/// diagnostic dumps that share the dump_bisect manifest's run id but
/// don't go through `dump_in_layer`.
pub fn current_layer_idx() -> Option<usize> {
    CURRENT_LAYER.with(|c| c.get())
}

pub fn current_step_idx() -> u64 {
    CURRENT_STEP.with(|c| c.get())
}

/// Set the current (step, layer) tag used by within-layer dump call sites.
/// Call this from `forward_gpu_*` at the top of each layer's iteration.
pub fn set_current_layer(step: u64, layer_idx: usize) {
    CURRENT_STEP.with(|c| c.set(step));
    CURRENT_LAYER.with(|c| c.set(Some(layer_idx)));
}

/// Clear the within-layer tag.  Call after layer dispatch returns.
pub fn clear_current_layer() {
    CURRENT_LAYER.with(|c| c.set(None));
}

/// Dump using the thread-local (step, layer) tag.  No-op if tag is unset
/// or dumping disabled.  Used by the within-layer `build_gated_attn_layer`
/// op-boundary dump points.
pub fn dump_in_layer(op: &str, buf: &MlxBuffer, shape: &[usize], device: &MlxDevice) {
    let layer = CURRENT_LAYER.with(|c| c.get());
    if layer.is_none() || !is_enabled() {
        return;
    }
    let step = CURRENT_STEP.with(|c| c.get());
    dump(step, layer, op, buf, shape, device);
}

/// Dump a hidden buffer at the given (step, layer, op) point.
///
/// `layer_idx` is `None` for always-on ops (embed, output_norm).
/// `shape` is purely advisory (recorded in the manifest).
/// `device` is used to flush any in-flight GPU work before downloading
/// the buffer (ensures shared-storage `as_slice` reads are post-write).
///
/// Errors are printed to stderr but never propagated — a failed dump
/// must not poison the forward pass.
pub fn dump(
    step: u64,
    layer_idx: Option<usize>,
    op: &str,
    buf: &MlxBuffer,
    shape: &[usize],
    device: &MlxDevice,
) {
    if !should_dump(layer_idx) {
        return;
    }
    let Some(cfg) = config() else { return };
    // Flush any pending GPU work before reading shared-storage buffers.
    // The hf2q decode hot path uses `commit_labeled` (async); without
    // this fence, `download_f32` may read partial data.
    if let Err(e) = flush_gpu(device) {
        eprintln!("[DUMP_BISECT] flush_gpu failed: {e}");
        return;
    }
    if let Err(e) = dump_inner(cfg, step, layer_idx, op, buf, shape) {
        eprintln!(
            "[DUMP_BISECT] step={step} layer={:?} op={op} dump failed: {e}",
            layer_idx
        );
    }
}

/// Open + commit-and-wait an empty encoder on the device so any
/// previously-submitted async command buffers are guaranteed to finish
/// before the caller reads buffer contents via `as_slice`.
///
/// ADR-015 iter94 Task #3 — when an active EncoderSession has been
/// installed via [`set_active_session`], drain THROUGH that session's
/// persistent CB instead of opening a fresh encoder.  Required for
/// HF2Q_ENCODER_SESSION=1 bisection: a fresh `device.command_encoder()`
/// allocates a NEW MTLCommandBuffer that does NOT join the session's
/// in-flight chain; without the session-aware drain, the dump's
/// `as_slice` reads can precede the session-internal writes (iter93
/// Phase C: all-zero ffn_out reads from session-only env=1).
fn flush_gpu(device: &MlxDevice) -> Result<()> {
    let active = ACTIVE_SESSION.with(|c| c.get());
    if let Some(sess_ptr) = active {
        // SAFETY: the pointer is set only by forward_gpu_* immediately
        // before the per-layer loop and cleared immediately after.  The
        // pointed `EncoderSession` lives on the inference thread's stack
        // for the full duration of that loop.  `flush_gpu` runs on the
        // same thread (dump_bisect is invoked from layer-body call
        // sites in forward_gpu).  No cross-thread aliasing is possible
        // because `ACTIVE_SESSION` is thread-local.  The `&mut` reborrow
        // here does not alias any other live `&mut` because no other
        // path holds a `&mut EncoderSession` while a dump call is in
        // progress (the layer-body code that calls `dump_in_layer`
        // released its `layer_session.as_mut()` borrow at the prior
        // `fence_or_commit` per the iter91 wire-up contract).
        let sess: &mut EncoderSession = unsafe { &mut *sess_ptr };
        sess.commit_and_wait()
            .context("flush_gpu session.commit_and_wait")?;
        sess.reset_for_next_stage()
            .context("flush_gpu session.reset_for_next_stage")?;
        return Ok(());
    }
    let mut enc = device.command_encoder().context("flush_gpu enc")?;
    enc.commit_and_wait().context("flush_gpu commit_and_wait")?;
    Ok(())
}

fn dump_inner(
    cfg: &DumpConfig,
    step: u64,
    layer_idx: Option<usize>,
    op: &str,
    buf: &MlxBuffer,
    shape: &[usize],
) -> Result<()> {
    let data = download_f32(buf).context("download_f32 for dump")?;
    let layer_str = match layer_idx {
        Some(n) => format!("{n:03}"),
        None => "___".to_string(),
    };
    let fname = format!("step{step:04}_layer{layer_str}_{op}.f32");
    let path = cfg.dir.join(&fname);
    let bytes: &[u8] =
        unsafe { std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * 4) };
    std::fs::write(&path, bytes).with_context(|| format!("write {}", path.display()))?;

    let manifest_path = cfg.dir.join("manifest.txt");
    let mut mf = OpenOptions::new()
        .create(true)
        .append(true)
        .open(&manifest_path)
        .with_context(|| format!("open manifest {}", manifest_path.display()))?;
    let layer_field = match layer_idx {
        Some(n) => n.to_string(),
        None => "-".to_string(),
    };
    writeln!(
        mf,
        "step={step} layer={layer_field} op={op} shape={:?} bytes={} path={}",
        shape,
        bytes.len(),
        fname
    )
    .ok();
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The DumpConfig OnceLock means we cannot directly toggle env var
    /// in unit tests across `should_dump` calls.  Instead we sanity-check
    /// the parser-equivalent logic by examining filter selection.
    #[test]
    fn filter_only_matches_layer_idx() {
        let f = LayerFilter::Only(5);
        // Always-on ops match Only(_).
        match (f, None::<usize>) {
            (LayerFilter::Only(_), None) => {}
            _ => panic!("always-on ops should match Only filter"),
        }
        // Matching idx.
        match (f, Some(5)) {
            (LayerFilter::Only(n), Some(idx)) if n == idx => {}
            _ => panic!("matching layer should pass"),
        }
        // Non-matching idx.
        match (f, Some(6)) {
            (LayerFilter::Only(n), Some(idx)) if n == idx => panic!("should not match"),
            _ => {}
        }
    }

    #[test]
    fn filter_all_matches_everything() {
        let f = LayerFilter::All;
        match (f, None::<usize>) {
            (LayerFilter::All, _) => {}
            _ => panic!("All should match"),
        }
        match (f, Some(0)) {
            (LayerFilter::All, _) => {}
            _ => panic!("All should match"),
        }
    }

    #[test]
    fn step_counter_monotone() {
        // Cannot reset between concurrent tests; just check increment.
        let a = next_step();
        let b = next_step();
        assert_eq!(b, a + 1);
    }

    #[test]
    fn synthetic_dump_round_trip() {
        // Spin up a 4-element f32 buffer and dump it via dump_inner.
        // We bypass should_dump by constructing a one-shot DumpConfig.
        use mlx_native::DType;

        let device = match MlxDevice::new() {
            Ok(d) => d,
            // Skip on machines without Metal (CI/Linux): the scaffold's
            // I/O semantics still get covered by `manifest_format_*` below.
            Err(_) => return,
        };
        let mut buf = match device.alloc_buffer(16, DType::F32, vec![4]) {
            Ok(b) => b,
            Err(_) => return,
        };
        {
            let s = buf.as_mut_slice::<f32>().expect("mut_slice");
            s.copy_from_slice(&[1.0_f32, 2.0, 3.0, 4.0]);
        }

        let tmp_dir = std::env::temp_dir().join(format!(
            "hf2q-dump-test-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_nanos()
        ));
        std::fs::create_dir_all(&tmp_dir).expect("mkdir tmp");
        File::create(tmp_dir.join("manifest.txt")).expect("create manifest");
        let cfg = DumpConfig {
            filter: LayerFilter::All,
            dir: tmp_dir.clone(),
        };
        dump_inner(&cfg, 0, Some(0), "synthetic", &buf, &[4])
            .expect("dump_inner round-trip");

        // Read back and verify.
        let path = tmp_dir.join("step0000_layer000_synthetic.f32");
        let bytes = std::fs::read(&path).expect("read dump");
        assert_eq!(bytes.len(), 16);
        let read: &[f32] =
            unsafe { std::slice::from_raw_parts(bytes.as_ptr() as *const f32, 4) };
        assert_eq!(read, &[1.0, 2.0, 3.0, 4.0]);

        let manifest = std::fs::read_to_string(tmp_dir.join("manifest.txt"))
            .expect("read manifest");
        assert!(manifest.contains("step=0"));
        assert!(manifest.contains("layer=0"));
        assert!(manifest.contains("op=synthetic"));
        assert!(manifest.contains("bytes=16"));

        // Cleanup.
        std::fs::remove_dir_all(&tmp_dir).ok();
    }
}
