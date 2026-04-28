//! Wave 5b.8 PP4096 measurement-spike instrumentation.
//!
//! Env gate: `HF2Q_PROFILE_W5B8=1` (default OFF — production codepath
//! unaffected when unset; matches the `HF2Q_DECODE_PROFILE` convention
//! at `forward_gpu.rs:698`).
//!
//! Purpose: capture per-section CPU wall-clock for the chunk-pipeline
//! prefill body so the W-5b.8 measurement spike (8.5× wall-clock gap
//! vs llama.cpp at pp4096) can be ranked by absolute ms contribution
//! instead of analytical guess. Zero kernel changes; no GPU
//! timestamping (per memory `project_m5max_no_dispatch_boundary_sampling`,
//! M5 Max only supports stage-boundary GPU counter sampling — CPU wall
//! captures encoder-split + commit_and_wait + memcpy + kernel-launch
//! overhead in a single number, which is what the gap is made of).
//!
//! Usage:
//! ```ignore
//! if w5b8_enabled() {
//!     let _t = Section::start(SectionKind::ChunkExpand);
//!     // ... work ...
//!     // _t drops -> recorded into thread-local accumulator
//! }
//! ```
//!
//! Print summary on demand via `w5b8_print_and_reset(label)` from
//! `forward_gpu_impl` after the per-layer loop completes.

use std::cell::RefCell;
use std::time::{Duration, Instant};

/// Sections instrumented in this measurement spike. Names map directly
/// to W-5b.8 task buckets — keep them stable so the docs section parses
/// the eprintln summary cleanly.
#[derive(Copy, Clone, Debug)]
pub enum SectionKind {
    /// One-time `upload_layer_weights_gpu` first-call cost.
    UploadWeights,
    /// `build_delta_net_layer` prefill ops1-3 encoder (pre_norm + qkv_proj
    /// + z_proj + ssm_conv + commit_and_wait). Per linear-attn layer.
    LayerOps1to3,
    /// CPU-side qkv_conv download + de-interleave + 3 GPU re-uploads
    /// (q_gpu, k_gpu, v_gpu). Per linear-attn layer.
    LayerQkvDeinterleave,
    /// Chunk-prep encoder (l2_norm q/k, alpha/beta proj, q_scale, g_beta
    /// + commit_and_wait). Per linear-attn layer, chunk path only.
    LayerChunkPrep,
    /// `apply_gated_delta_net_chunk` total wall (= sum of the four
    /// sub-buckets below + return-value memcpy). Per linear-attn layer.
    LayerChunkCall,
    /// Inside chunk wrapper: GQA F32 expansion CPU memcpy
    /// (`q_expanded`/`k_expanded` allocation + tiled fill loop, lines
    /// 880-933 of `gpu_delta_net.rs`).
    ChunkGqaExpand,
    /// Inside chunk wrapper: scratch BF16 + g_log_decay + final_state +
    /// output buffer allocations.
    ChunkAllocs,
    /// Inside chunk wrapper: wall around the single mega-encoder build
    /// (cast → sign-flip → `dispatch_chunk_gated_delta_rule_fwd` → cast
    /// back) — i.e. encoder build time without the commit.
    ChunkEncBuild,
    /// Inside chunk wrapper: the `enc.commit_and_wait()` at the very end
    /// of `apply_gated_delta_net_chunk` — measures GPU wait time for the
    /// 6-kernel chunk pipeline + 4 casts.
    ChunkCommitWait,
    /// Chunk-ops8-9 encoder (ssm_norm_gate + out_proj + commit_and_wait).
    /// Per linear-attn layer, chunk path only.
    LayerChunkOps8to9,
    /// Autoregressive ops5-9 encoder (l2_norm + alpha/beta proj +
    /// q_scale + g_beta + GDN + ssm_norm_gate + out_proj +
    /// commit_and_wait). Per linear-attn layer, autoreg path only.
    LayerAutoregOps5to9,
    /// Per linear-attn layer total wall (== sum of all per-layer buckets
    /// for that path; the residual after subtraction reveals
    /// non-instrumented overhead).
    LayerLinearTotal,
    /// Per full-attn layer total wall — includes the gated-attn
    /// build_gated_attn_layer call (no chunk pipeline involved).
    LayerFullTotal,
}

impl SectionKind {
    fn idx(self) -> usize {
        self as usize
    }
    fn label(self) -> &'static str {
        match self {
            SectionKind::UploadWeights => "upload_weights",
            SectionKind::LayerOps1to3 => "layer.ops1_3",
            SectionKind::LayerQkvDeinterleave => "layer.qkv_deinterleave",
            SectionKind::LayerChunkPrep => "layer.chunk_prep",
            SectionKind::LayerChunkCall => "layer.chunk_call",
            SectionKind::ChunkGqaExpand => "chunk.gqa_expand",
            SectionKind::ChunkAllocs => "chunk.allocs",
            SectionKind::ChunkEncBuild => "chunk.enc_build",
            SectionKind::ChunkCommitWait => "chunk.commit_wait",
            SectionKind::LayerChunkOps8to9 => "layer.chunk_ops8_9",
            SectionKind::LayerAutoregOps5to9 => "layer.autoreg_ops5_9",
            SectionKind::LayerLinearTotal => "layer.linear_total",
            SectionKind::LayerFullTotal => "layer.full_total",
        }
    }
    const COUNT: usize = 13;
}

#[derive(Default, Clone)]
struct Acc {
    samples: Vec<u128>, // microseconds per sample; small N (≤64 layers)
}

impl Acc {
    fn record(&mut self, dur: Duration) {
        self.samples.push(dur.as_micros());
    }
    fn count(&self) -> usize {
        self.samples.len()
    }
    fn sum_us(&self) -> u128 {
        self.samples.iter().sum()
    }
    fn min_us(&self) -> u128 {
        self.samples.iter().copied().min().unwrap_or(0)
    }
    fn max_us(&self) -> u128 {
        self.samples.iter().copied().max().unwrap_or(0)
    }
    fn mean_us(&self) -> u128 {
        if self.samples.is_empty() {
            0
        } else {
            self.sum_us() / self.samples.len() as u128
        }
    }
    fn percentile_us(&self, p: f64) -> u128 {
        if self.samples.is_empty() {
            return 0;
        }
        let mut s = self.samples.clone();
        s.sort_unstable();
        let idx = ((s.len() - 1) as f64 * p).round() as usize;
        s[idx]
    }
}

#[derive(Default, Clone)]
struct W5b8State {
    accs: Vec<Acc>,
}

impl W5b8State {
    fn new() -> Self {
        Self {
            accs: vec![Acc::default(); SectionKind::COUNT],
        }
    }
    fn record(&mut self, kind: SectionKind, dur: Duration) {
        if self.accs.is_empty() {
            self.accs = vec![Acc::default(); SectionKind::COUNT];
        }
        self.accs[kind.idx()].record(dur);
    }
}

thread_local! {
    static W5B8: RefCell<W5b8State> = RefCell::new(W5b8State::new());
}

/// True when `HF2Q_PROFILE_W5B8=1` is set in the environment.
#[inline]
pub fn w5b8_enabled() -> bool {
    // Cheap getenv every call is fine — measurement-spike code, not hot
    // path; matches the existing `HF2Q_DECODE_PROFILE` pattern at
    // `forward_gpu.rs:698`.
    std::env::var("HF2Q_PROFILE_W5B8").is_ok()
}

/// RAII guard. `Section::start(kind)` records the elapsed wall-clock
/// into the thread-local accumulator on drop; no-ops when the env gate
/// is off (the constructor still allocates an Instant, but that's a
/// few-ns rdtsc — negligible vs measurement noise).
pub struct Section {
    kind: SectionKind,
    t0: Option<Instant>,
}

impl Section {
    pub fn start(kind: SectionKind) -> Self {
        let t0 = if w5b8_enabled() { Some(Instant::now()) } else { None };
        Self { kind, t0 }
    }
}

impl Drop for Section {
    fn drop(&mut self) {
        if let Some(t0) = self.t0 {
            let dur = t0.elapsed();
            W5B8.with(|cell| cell.borrow_mut().record(self.kind, dur));
        }
    }
}

/// Print a one-shot summary of all accumulated buckets to stderr and
/// reset the thread-local state. Called from `forward_gpu_impl` after
/// the per-layer loop completes (gated on `w5b8_enabled()`).
pub fn w5b8_print_and_reset(label: &str) {
    if !w5b8_enabled() {
        return;
    }
    W5B8.with(|cell| {
        let mut state = cell.borrow_mut();
        eprintln!("[W5B8_PROFILE] === section summary: {label} ===");
        eprintln!(
            "[W5B8_PROFILE] {:<26} {:>6} {:>10} {:>10} {:>10} {:>10} {:>10} {:>10}",
            "section", "n", "sum_ms", "mean_ms", "min_ms", "max_ms", "p50_ms", "p95_ms"
        );
        // Iterate in the SectionKind order for stable output.
        let kinds = [
            SectionKind::UploadWeights,
            SectionKind::LayerOps1to3,
            SectionKind::LayerQkvDeinterleave,
            SectionKind::LayerChunkPrep,
            SectionKind::LayerChunkCall,
            SectionKind::ChunkGqaExpand,
            SectionKind::ChunkAllocs,
            SectionKind::ChunkEncBuild,
            SectionKind::ChunkCommitWait,
            SectionKind::LayerChunkOps8to9,
            SectionKind::LayerAutoregOps5to9,
            SectionKind::LayerLinearTotal,
            SectionKind::LayerFullTotal,
        ];
        for k in kinds {
            let acc = &state.accs[k.idx()];
            if acc.count() == 0 {
                continue;
            }
            eprintln!(
                "[W5B8_PROFILE] {:<26} {:>6} {:>10.3} {:>10.3} {:>10.3} {:>10.3} {:>10.3} {:>10.3}",
                k.label(),
                acc.count(),
                acc.sum_us() as f64 / 1000.0,
                acc.mean_us() as f64 / 1000.0,
                acc.min_us() as f64 / 1000.0,
                acc.max_us() as f64 / 1000.0,
                acc.percentile_us(0.50) as f64 / 1000.0,
                acc.percentile_us(0.95) as f64 / 1000.0,
            );
        }
        eprintln!("[W5B8_PROFILE] === end summary ===");
        // Reset for subsequent calls (the binary issues one forward at
        // pp4096 then exits, but unit tests may iterate).
        *state = W5b8State::new();
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn section_records_only_when_enabled() {
        // gate off (default) — record() no-op
        std::env::remove_var("HF2Q_PROFILE_W5B8");
        {
            let _t = Section::start(SectionKind::UploadWeights);
            std::thread::sleep(Duration::from_millis(1));
        }
        W5B8.with(|cell| {
            let s = cell.borrow();
            assert_eq!(
                s.accs[SectionKind::UploadWeights.idx()].count(),
                0,
                "section should not record when env unset"
            );
        });

        // gate on — record() captures one sample
        std::env::set_var("HF2Q_PROFILE_W5B8", "1");
        {
            let _t = Section::start(SectionKind::UploadWeights);
            std::thread::sleep(Duration::from_millis(2));
        }
        W5B8.with(|cell| {
            let s = cell.borrow();
            assert_eq!(s.accs[SectionKind::UploadWeights.idx()].count(), 1);
            assert!(s.accs[SectionKind::UploadWeights.idx()].sum_us() >= 1_000);
        });
        // Cleanup so other tests don't see the env var.
        std::env::remove_var("HF2Q_PROFILE_W5B8");
        // Reset thread-local for downstream test isolation.
        W5B8.with(|cell| *cell.borrow_mut() = W5b8State::new());
    }
}
