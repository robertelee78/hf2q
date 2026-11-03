//! ADR-063 Q4-GEMMA warm-prefill category split (`HF2Q_GEMMA_PREFILL_CATSPLIT=1`).
//!
//! MEASUREMENT-ONLY and off by default. The Gemma 4 warm (live-prefix resume)
//! batched prefill normally commits one asynchronous command buffer per layer.
//! With the gate on, each layer of a warm forward is instead split at four
//! boundaries (before live attention, after it, before the routed MoE, and at
//! the layer end). Each closed segment is committed with
//! `finish_with_gpu_time` (the command buffer's `GPUEndTime - GPUStartTime`)
//! and its GPU-busy time is added to that segment's category. The dispatches,
//! kernels and their order are unchanged; only command-buffer boundaries and
//! CPU waits are added, so the serialized segment sum overstates the real
//! overlapped layer time. Report it as a split of GPU-busy time, and measure the
//! perturbation as profiled versus unprofiled makespan on the same workload.
//!
//! Not covered: cold (non-resume) prefill, the prefill setup (embedding/masks)
//! before layer 0, the head after the last layer, resume suffixes under 32
//! tokens (replayed through scalar decode), and `HF2Q_GRAPH_OPT_PREFILL=1`
//! (recorded sessions; the boundaries are skipped there). Combining this gate
//! with the `HF2Q_PROFILE_*` bucket gates drops time from these categories,
//! because those gates close sessions without GPU timestamps.
//!
//! The Gemma 4 worker prints one `[PREFILL_CATSPLIT_JSON]` line of lifetime
//! totals at worker exit (graceful shutdown), next to `[CATSPLIT]`.
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::LazyLock;

use anyhow::Result;
use mlx_native::graph::{GraphExecutor, GraphSession};

pub(crate) const ENV: &str = "HF2Q_GEMMA_PREFILL_CATSPLIT";
pub(crate) const SCHEMA: &str = "hf2q.gemma4.prefill-catsplit.v1";

/// Segment category; the value is the bucket index and the report order.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Cat {
    /// Input norm, Q/K/V projections, head norms, RoPE and KV writes.
    PreAttn = 0,
    /// The live warm attention (`encode_live_layer_attention`).
    Attention = 1,
    /// Everything after attention up to the routed experts: O projection,
    /// post-attention norm/residual, the dense FFN, the router and the routed
    /// experts' input norm.
    Dense = 2,
    /// Routed experts and the layer tail: gate/up, SwiGLU, down, weighted sum,
    /// end-of-layer norm/residual.
    Moe = 3,
}

pub(crate) const LEN: usize = 4;
pub(crate) const NAMES: [&str; LEN] = ["pre_attn", "attention", "dense", "moe"];

static NS: [AtomicU64; LEN] = [const { AtomicU64::new(0) }; LEN];
static CBS: [AtomicU64; LEN] = [const { AtomicU64::new(0) }; LEN];
static FORWARDS: AtomicU64 = AtomicU64::new(0);
static ROWS: AtomicU64 = AtomicU64::new(0);
static LANES: AtomicU64 = AtomicU64::new(0);

static ENABLED: LazyLock<bool> = LazyLock::new(|| enabled_value(std::env::var(ENV).ok().as_deref()));

fn enabled_value(value: Option<&str>) -> bool {
    value == Some("1")
}

pub(crate) fn enabled() -> bool {
    *ENABLED
}

/// Count one warm forward of `rows` total rows over `lanes` sequences.
pub(crate) fn forward(rows: usize, lanes: usize) {
    FORWARDS.fetch_add(1, Ordering::Relaxed);
    ROWS.fetch_add(rows as u64, Ordering::Relaxed);
    LANES.fetch_add(lanes as u64, Ordering::Relaxed);
}

fn add(c: Cat, ns: u64) {
    NS[c as usize].fetch_add(ns, Ordering::Relaxed);
    CBS[c as usize].fetch_add(1, Ordering::Relaxed);
}

/// Close the live session's segment as category `c` and continue in a fresh
/// session. Callers gate it; `c` labels the work encoded since the previous
/// boundary or session begin. The finish is a full GPU sync, so later
/// segments observe every earlier write.
pub(crate) fn boundary<'a>(
    session: &mut GraphSession<'a>,
    exec: &'a GraphExecutor,
    c: Cat,
) -> Result<()> {
    let fresh = exec
        .begin()
        .map(crate::inference::native_observation::session)
        .map_err(|e| anyhow::anyhow!("prefill catsplit re-begin: {e}"))?;
    let ns = std::mem::replace(session, fresh)
        .finish_with_gpu_time()
        .map_err(|e| anyhow::anyhow!("prefill catsplit finish: {e}"))?;
    add(c, ns);
    Ok(())
}

#[derive(Debug, PartialEq, Eq)]
pub(crate) struct Snapshot {
    pub ns: [u64; LEN],
    pub cbs: [u64; LEN],
    pub forwards: u64,
    pub rows: u64,
    pub lanes: u64,
}

pub(crate) fn snapshot() -> Snapshot {
    Snapshot {
        ns: std::array::from_fn(|i| NS[i].load(Ordering::Relaxed)),
        cbs: std::array::from_fn(|i| CBS[i].load(Ordering::Relaxed)),
        forwards: FORWARDS.load(Ordering::Relaxed),
        rows: ROWS.load(Ordering::Relaxed),
        lanes: LANES.load(Ordering::Relaxed),
    }
}

/// The machine-readable worker-exit record.
pub(crate) fn report(s: &Snapshot) -> serde_json::Value {
    let categories: serde_json::Map<String, serde_json::Value> = NAMES
        .iter()
        .enumerate()
        .map(|(i, name)| {
            (
                (*name).to_owned(),
                serde_json::json!({"gpu_ns": s.ns[i], "command_buffers": s.cbs[i]}),
            )
        })
        .collect();
    serde_json::json!({
        "schema": SCHEMA,
        "scope": "gemma4 warm live-prefix-resume batched prefill, layers only, worker lifetime",
        "clock": "MTLCommandBuffer GPUEndTime-GPUStartTime per serialized segment",
        "forwards": s.forwards,
        "rows": s.rows,
        "lanes": s.lanes,
        "categories": categories,
        "gpu_ns_total": s.ns.iter().sum::<u64>(),
    })
}

/// Print the lifetime totals once at Gemma 4 worker exit; no-op when off.
pub(crate) fn emit_worker_exit() {
    if enabled() {
        eprintln!("[PREFILL_CATSPLIT_JSON] {}", report(&snapshot()));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gate_requires_exact_one() {
        assert!(enabled_value(Some("1")));
        for v in [None, Some("0"), Some(""), Some("true"), Some(" 1")] {
            assert!(!enabled_value(v), "{v:?}");
        }
    }

    #[test]
    fn names_follow_category_indices() {
        assert_eq!(NAMES[Cat::PreAttn as usize], "pre_attn");
        assert_eq!(NAMES[Cat::Attention as usize], "attention");
        assert_eq!(NAMES[Cat::Dense as usize], "dense");
        assert_eq!(NAMES[Cat::Moe as usize], "moe");
    }

    #[test]
    fn report_carries_every_category_and_totals() {
        let s = Snapshot {
            ns: [10, 200, 30, 400],
            cbs: [1, 2, 3, 4],
            forwards: 5,
            rows: 600,
            lanes: 7,
        };
        let r = report(&s);
        assert_eq!(r["schema"], SCHEMA);
        assert_eq!(r["gpu_ns_total"], 640);
        assert_eq!(r["forwards"], 5);
        assert_eq!(r["rows"], 600);
        assert_eq!(r["lanes"], 7);
        assert_eq!(r["categories"]["attention"]["gpu_ns"], 200);
        assert_eq!(r["categories"]["moe"]["command_buffers"], 4);
        assert_eq!(r["categories"].as_object().unwrap().len(), LEN);
    }

    #[test]
    fn accumulators_add_per_category() {
        // Global counters: assert deltas so parallel tests stay independent.
        let before = snapshot();
        add(Cat::Attention, 25);
        add(Cat::Attention, 5);
        forward(33, 2);
        let after = snapshot();
        assert!(after.ns[Cat::Attention as usize] >= before.ns[Cat::Attention as usize] + 30);
        assert!(after.cbs[Cat::Attention as usize] >= before.cbs[Cat::Attention as usize] + 2);
        assert!(after.forwards > before.forwards);
        assert!(after.rows >= before.rows + 33);
        assert!(after.lanes >= before.lanes + 2);
    }
}
