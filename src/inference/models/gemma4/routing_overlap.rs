//! ADR-063 Q4-GEMMA-DECODE V2 workload diagnostic (`HF2Q_GEMMA_ROUTING_OVERLAP=1`).
//!
//! DIAGNOSTIC-ONLY and off by default; the gate is read once per process and
//! costs one cached load per decode step when off. With the gate on, the
//! `[N,hidden]` batched decode body blit-copies each layer's `moe_expert_ids`
//! into a per-step `[layers, N, top_k]` u32 buffer and reads it after the
//! step's existing final wait (no added synchronization). Per step it also
//! records each slot's KV length per layer type and whether the step was
//! mixed-bucket, using the same `(nwg, nsg)` bucket rule as the batched flash
//! gate in `encode_one_layer_batched`. The Gemma 4 worker prints one
//! `[ROUTING_OVERLAP]` JSON line of lifetime histograms at worker exit:
//! distinct experts `U` per (layer, width), KV length per layer type, and the
//! mixed-bucket share (design §7.4).
use std::collections::BTreeMap;
use std::sync::{LazyLock, Mutex};

pub(crate) const ENV: &str = "HF2Q_GEMMA_ROUTING_OVERLAP";
pub(crate) const SCHEMA: &str = "hf2q.gemma4.routing-overlap.v1";

static ENABLED: LazyLock<bool> =
    LazyLock::new(|| enabled_value(std::env::var(ENV).ok().as_deref()));

fn enabled_value(value: Option<&str>) -> bool {
    value == Some("1")
}

pub(crate) fn enabled() -> bool {
    *ENABLED
}

/// KV length of one slot at one layer, as the batched attention computes it:
/// rings count `min(ring_pos + 1, capacity)` rows where `ring_pos` is the
/// position minus `ring_offset` (the hybrid regime's graft length; `0` for
/// full-TQ), full layers count `seq_pos + 1` rows.
pub(crate) fn kv_len(seq_pos: usize, capacity: usize, is_ring: bool, ring_offset: usize) -> u32 {
    if is_ring {
        ((seq_pos.saturating_sub(ring_offset) + 1).min(capacity)) as u32
    } else {
        (seq_pos + 1) as u32
    }
}

fn nwg_bucket(k: u32) -> u32 {
    if k > 512 {
        32
    } else {
        16
    }
}

fn nsg_of(k: u32) -> u32 {
    mlx_native::ops::flash_attn_vec_tq_hb::compute_nsg(k)
}

/// True when every slot shares the widest slot's `(nwg, nsg)` bucket — the
/// condition under which the batched flash runs instead of the per-slot
/// fallback.
pub(crate) fn same_bucket(kv_lens: &[u32]) -> bool {
    let max = kv_lens.iter().copied().max().unwrap_or(1);
    let nsg_max = nsg_of(max);
    kv_lens
        .iter()
        .all(|&k| nwg_bucket(k) == nwg_bucket(max) && nsg_of(k) == nsg_max)
}

/// Number of distinct expert ids in one layer's `[n, top_k]` selection.
pub(crate) fn distinct_experts(layer_ids: &[u32], n: usize, stride: usize, top_k: usize) -> u32 {
    let mut seen: Vec<u32> = (0..n)
        .flat_map(|i| layer_ids[i * stride..i * stride + top_k].iter().copied())
        .collect();
    seen.sort_unstable();
    seen.dedup();
    seen.len() as u32
}

/// One layer of one recorded step.
pub(crate) struct StepLayer {
    pub sliding: bool,
    pub top_k: usize,
    pub kv_lens: Vec<u32>,
}

#[derive(Default)]
struct TypeStats {
    /// Power-of-two upper bound of each slot's KV length -> count.
    kv_pow2: BTreeMap<u32, u64>,
    /// `(nwg, nsg)` attention bucket of each slot's KV length -> count.
    kv_bucket: BTreeMap<(u32, u32), u64>,
    layer_steps: u64,
    mixed_layer_steps: u64,
}

#[derive(Default)]
struct State {
    steps: u64,
    mixed_steps: u64,
    steps_by_width: BTreeMap<usize, u64>,
    /// (layer, width) -> (top_k, U -> count).
    distinct: BTreeMap<(usize, usize), (usize, BTreeMap<u32, u64>)>,
    sliding: TypeStats,
    global: TypeStats,
}

static STATE: LazyLock<Mutex<State>> = LazyLock::new(|| Mutex::new(State::default()));

/// Fold one decode step. `ids` is the `[layers.len(), n, stride]` copy of the
/// per-layer `moe_expert_ids`.
pub(crate) fn record_step(ids: &[u32], n: usize, stride: usize, layers: &[StepLayer]) {
    let mut st = STATE.lock().unwrap_or_else(|e| e.into_inner());
    record_into(&mut st, ids, n, stride, layers);
}

fn record_into(st: &mut State, ids: &[u32], n: usize, stride: usize, layers: &[StepLayer]) {
    st.steps += 1;
    *st.steps_by_width.entry(n).or_default() += 1;
    let mut any_mixed = false;
    let (mut kv_sliding_done, mut kv_global_done) = (false, false);
    for (layer, l) in layers.iter().enumerate() {
        let layer_ids = &ids[layer * n * stride..(layer + 1) * n * stride];
        let u = distinct_experts(layer_ids, n, stride, l.top_k);
        let entry = st
            .distinct
            .entry((layer, n))
            .or_insert_with(|| (l.top_k, BTreeMap::new()));
        *entry.1.entry(u).or_default() += 1;

        let (stats, kv_done) = if l.sliding {
            (&mut st.sliding, &mut kv_sliding_done)
        } else {
            (&mut st.global, &mut kv_global_done)
        };
        let mixed = !same_bucket(&l.kv_lens);
        any_mixed |= mixed;
        stats.layer_steps += 1;
        stats.mixed_layer_steps += mixed as u64;
        // KV lengths once per layer type per step (every layer of a type
        // shares capacity), so the histogram counts slot-steps.
        if !*kv_done {
            *kv_done = true;
            for &k in &l.kv_lens {
                *stats.kv_pow2.entry(k.max(1).next_power_of_two()).or_default() += 1;
                *stats.kv_bucket.entry((nwg_bucket(k), nsg_of(k))).or_default() += 1;
            }
        }
    }
    st.mixed_steps += any_mixed as u64;
}

fn share(part: u64, whole: u64) -> f64 {
    if whole == 0 {
        0.0
    } else {
        part as f64 / whole as f64
    }
}

fn type_report(t: &TypeStats) -> serde_json::Value {
    let pow2: serde_json::Map<String, serde_json::Value> = t
        .kv_pow2
        .iter()
        .map(|(k, v)| (format!("le{k}"), (*v).into()))
        .collect();
    let bucket: serde_json::Map<String, serde_json::Value> = t
        .kv_bucket
        .iter()
        .map(|((nwg, nsg), v)| (format!("nwg{nwg}_nsg{nsg}"), (*v).into()))
        .collect();
    serde_json::json!({
        "kv_len_pow2": pow2,
        "kv_len_bucket": bucket,
        "layer_steps": t.layer_steps,
        "mixed_layer_steps": t.mixed_layer_steps,
        "mixed_share": share(t.mixed_layer_steps, t.layer_steps),
    })
}

fn report_of(st: &State) -> serde_json::Value {
    let widths: serde_json::Map<String, serde_json::Value> = st
        .steps_by_width
        .iter()
        .map(|(w, c)| (w.to_string(), (*c).into()))
        .collect();
    let distinct: Vec<serde_json::Value> = st
        .distinct
        .iter()
        .map(|((layer, width), (top_k, hist))| {
            let h: serde_json::Map<String, serde_json::Value> =
                hist.iter().map(|(u, c)| (u.to_string(), (*c).into())).collect();
            serde_json::json!({"layer": layer, "width": width, "top_k": top_k, "u_hist": h})
        })
        .collect();
    serde_json::json!({
        "schema": SCHEMA,
        "scope": "gemma4 [N,hidden] batched decode body, worker lifetime",
        "steps": st.steps,
        "steps_by_width": widths,
        "mixed_bucket_steps": st.mixed_steps,
        "mixed_bucket_share": share(st.mixed_steps, st.steps),
        "distinct_experts": distinct,
        "sliding": type_report(&st.sliding),
        "global": type_report(&st.global),
    })
}

/// Print the lifetime histograms once at Gemma 4 worker exit; no-op when off.
pub(crate) fn emit_worker_exit() {
    if enabled() {
        let st = STATE.lock().unwrap_or_else(|e| e.into_inner());
        eprintln!("[ROUTING_OVERLAP] {}", report_of(&st));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn routing_overlap_gate_requires_exact_one() {
        assert!(enabled_value(Some("1")));
        for v in [None, Some("0"), Some(""), Some("true"), Some(" 1")] {
            assert!(!enabled_value(v), "{v:?}");
        }
    }

    #[test]
    fn routing_overlap_kv_len_matches_attention_rule() {
        assert_eq!(kv_len(9, 1024, false, 0), 10);
        assert_eq!(kv_len(2000, 1024, true, 0), 1024);
        assert_eq!(kv_len(20, 1024, true, 5), 16);
        assert_eq!(kv_len(3, 1024, true, 10), 1);
        assert_eq!(kv_len(30, 1024, false, 10), 31, "full layers ignore the ring offset");
    }

    #[test]
    fn routing_overlap_counts_distinct_experts_per_layer() {
        // n = 2, stride 3, top_k 2: slot rows [1, 2, x] and [2, 5, x].
        let ids = [1, 2, 99, 2, 5, 99];
        assert_eq!(distinct_experts(&ids, 2, 3, 2), 3);
        assert_eq!(distinct_experts(&[4, 4, 4, 4], 2, 2, 2), 1);
    }

    #[test]
    fn routing_overlap_bucket_split_is_mixed() {
        assert!(same_bucket(&[100, 200, 300, 400]));
        assert!(!same_bucket(&[100, 4000]), "nwg 16 vs 32");
    }

    #[test]
    fn routing_overlap_report_carries_histograms() {
        let mut st = State::default();
        // Two layers (sliding, global), n = 2, top_k = 2, stride 2.
        let layers = [
            StepLayer { sliding: true, top_k: 2, kv_lens: vec![10, 20] },
            StepLayer { sliding: false, top_k: 2, kv_lens: vec![100, 4000] },
        ];
        let ids = [0, 1, 1, 2, 3, 3, 3, 3];
        record_into(&mut st, &ids, 2, 2, &layers);
        record_into(&mut st, &ids, 2, 2, &layers[..]);
        let r = report_of(&st);
        assert_eq!(r["schema"], SCHEMA);
        assert_eq!(r["steps"], 2);
        assert_eq!(r["steps_by_width"]["2"], 2);
        assert_eq!(r["mixed_bucket_steps"], 2);
        assert_eq!(r["mixed_bucket_share"], 1.0);
        assert_eq!(r["distinct_experts"][0]["layer"], 0);
        assert_eq!(r["distinct_experts"][0]["u_hist"]["3"], 2);
        assert_eq!(r["distinct_experts"][1]["u_hist"]["1"], 2);
        assert_eq!(r["sliding"]["kv_len_pow2"]["le16"], 2);
        assert_eq!(r["sliding"]["kv_len_pow2"]["le32"], 2);
        assert_eq!(r["sliding"]["mixed_share"], 0.0);
        assert_eq!(r["global"]["mixed_layer_steps"], 2);
        assert_eq!(r["global"]["kv_len_bucket"].as_object().unwrap().len(), 2);
    }
}
