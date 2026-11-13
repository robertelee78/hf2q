//! Deep-context gate of the Gemma 4 warm attention default (ADR-063
//! Q4-GEMMA-ATTN, v0.1.30): on the official GGUF, a warm resume at depth on
//! the tensor route with bounded global-layer scratch
//!
//! * is bit for bit the unbounded route (first token, logits, every K/V
//!   row), for one lane and for a joint cohort, and
//! * keeps its accounted scratch inside the documented envelope
//!   (`max(budget, one sequence's dense K/V) + one mask`).
//!
//! It also prints wall time and peak transient Metal allocation of the
//! vec-hybrid (kill switch), bounded tensor (default) and unbounded tensor
//! arms, one `GEMMA_WARM_DEPTH ARM` line each.
//!
//! ```text
//! HF2Q_GEMMA4_GGUF=/abs/gemma-4-26B-A4B-it-hf2q-q4_k_m.gguf \
//! HF2Q_GEMMA_WARM_DEPTH_TOKENS=102400 HF2Q_GEMMA_WARM_DEPTH_LANES=1 \
//!   cargo test --release --locked --bin hf2q \
//!   gemma_warm_tensor_scratch_is_bounded_and_bitwise_at_depth -- --ignored --nocapture
//! ```
//!
//! `HF2Q_GEMMA_WARM_DEPTH_TOKENS` (default 32768) is each lane's history,
//! `HF2Q_GEMMA_WARM_DEPTH_ROWS` (default 2048) its suffix,
//! `HF2Q_GEMMA_WARM_DEPTH_LANES` (default 1, up to 4) the cohort, and
//! `HF2Q_GEMMA_WARM_DEPTH_ARMS` (default `vec,bounded,unbounded`) the arms;
//! the bitwise comparison needs both tensor arms.
use super::gemma_warm_batch_tests::{
    assert_lane_equal, lane_image, load_gate_model, synthetic_prompt, LaneImage, MAX_DECODE,
};
use super::*;
use crate::inference::models::gemma4::kv_cache::{
    restore_gemma_hybrid_slot_anchor, snapshot_gemma_hybrid_slot_anchor, GemmaHybridSlotAnchor,
};
use crate::inference::models::gemma4::GemmaLiveKernel;
use crate::serve::gemma_live_attention::{
    last_forward_scratch_stats, set_test_tensor_scratch_limits, LiveScratchStats,
    TensorScratchLimits, TENSOR_GLOBAL_MASK_BYTES_MAX, TENSOR_GLOBAL_SCRATCH_BUDGET_BYTES,
};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

const BUILD_STEP: usize = 4_096;
const MIB: u64 = 1 << 20;
/// Dense F16 K and V bytes per key of one 26B-A4B global layer (2 KV heads
/// of head dim 512).
const GLOBAL_DENSE_BYTES_PER_KEY: u64 = 4_096;

fn env_usize(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .map(|value| value.trim().parse().unwrap_or_else(|_| panic!("{name}")))
        .unwrap_or(default)
}

fn allocated(guard: &Gemma4KvGuard<'_>) -> u64 {
    guard
        .model
        .ctx
        .device()
        .metal_device()
        .current_allocated_size() as u64
}

fn resume(guard: &mut Gemma4KvGuard<'_>, slot: usize, prompt: &[u32], history: usize) -> u32 {
    let first = guard
        .model
        .weights
        .forward_prefill_with_soft_tokens_slot_aware_resume(
            prompt,
            &[],
            MAX_DECODE,
            &mut guard.model.ctx,
            SlotId(slot as u32),
            &mut guard.kv,
            guard.hybrid.as_mut(),
            guard.dense.as_mut(),
            guard.mlx.as_mut(),
            history,
        )
        .unwrap();
    clear_gemma4_self_mounts(guard.model);
    commit_gemma4_slot_cursor(guard, SlotId(slot as u32), history, prompt.len()).unwrap();
    first
}

struct Arm {
    label: &'static str,
    wall: Duration,
    peak_extra: u64,
    stats: LiveScratchStats,
    lanes: Vec<LaneImage>,
}

#[test]
#[ignore = "official Gemma 4 GGUF: warm tensor attention at depth is bit-identical to the unbounded route and its global-layer scratch stays in the envelope"]
fn gemma_warm_tensor_scratch_is_bounded_and_bitwise_at_depth() {
    let depth = env_usize("HF2Q_GEMMA_WARM_DEPTH_TOKENS", 32_768);
    let rows = env_usize("HF2Q_GEMMA_WARM_DEPTH_ROWS", 2_048);
    let lanes = env_usize("HF2Q_GEMMA_WARM_DEPTH_LANES", 1);
    let arms_env = std::env::var("HF2Q_GEMMA_WARM_DEPTH_ARMS")
        .unwrap_or_else(|_| "vec,bounded,unbounded".to_string());
    assert!((1..=4).contains(&lanes) && rows >= 32 && lanes * rows <= 4_096);
    assert!(depth >= BUILD_STEP);

    let started = Instant::now();
    let mut loaded = load_gate_model();
    let LoadedModel::Gemma(g) = &mut loaded else {
        unreachable!("load_gate_model returns Gemma")
    };
    let mut guard = Gemma4KvGuard::take(g).unwrap();
    let vocab = guard.model.weights.vocab_size;
    let prompts: Vec<Vec<u32>> = (0..lanes)
        .map(|slot| synthetic_prompt(slot, depth + rows, vocab))
        .collect();

    // Each lane's history: one cold prefill, then warm resumes on the
    // default route (tensor, serving limits).
    guard.model.weights.warm_live_kernel = GemmaLiveKernel::TensorF16;
    set_test_tensor_scratch_limits(None);
    let mut anchors: Vec<GemmaHybridSlotAnchor> = Vec::new();
    for slot in 0..lanes {
        let mut cursor = BUILD_STEP.min(depth);
        guard
            .model
            .weights
            .forward_prefill_with_soft_tokens_slot_aware(
                &prompts[slot][..cursor],
                &[],
                MAX_DECODE,
                &mut guard.model.ctx,
                SlotId(slot as u32),
                &mut guard.kv,
                guard.hybrid.as_mut(),
                guard.dense.as_mut(),
                guard.mlx.as_mut(),
            )
            .unwrap();
        clear_gemma4_self_mounts(guard.model);
        install_gemma4_slot_cursor(&mut guard, SlotId(slot as u32), cursor).unwrap();
        while cursor < depth {
            let next = (cursor + BUILD_STEP).min(depth);
            resume(&mut guard, slot, &prompts[slot][..next], cursor);
            cursor = next;
        }
        anchors.push(
            snapshot_gemma_hybrid_slot_anchor(
                guard.hybrid.as_ref().unwrap(),
                SlotId(slot as u32),
                depth,
            )
            .unwrap(),
        );
    }
    eprintln!(
        "GEMMA_WARM_DEPTH built lanes={lanes} history={depth} in {:.1}s; metal allocated {} MiB",
        started.elapsed().as_secs_f64(),
        allocated(&guard) / MIB
    );

    let mut arms: Vec<Arm> = Vec::new();
    for (label, kernel, limits) in [
        (
            "vec",
            GemmaLiveKernel::VecHybrid,
            TensorScratchLimits::SERVING,
        ),
        (
            "bounded",
            GemmaLiveKernel::TensorF16,
            TensorScratchLimits::SERVING,
        ),
        (
            "unbounded",
            GemmaLiveKernel::TensorF16,
            TensorScratchLimits::UNBOUNDED,
        ),
    ] {
        if !arms_env.split(',').any(|arm| arm.trim() == label) {
            continue;
        }
        for slot in 0..lanes {
            restore_gemma_hybrid_slot_anchor(
                guard.hybrid.as_mut().unwrap(),
                SlotId(slot as u32),
                &anchors[slot],
                depth,
            )
            .unwrap();
            install_gemma4_slot_cursor(&mut guard, SlotId(slot as u32), depth).unwrap();
        }
        guard.model.weights.warm_live_kernel = kernel;
        set_test_tensor_scratch_limits(Some(limits));
        let base = allocated(&guard);
        let stop = Arc::new(AtomicBool::new(false));
        let peak = Arc::new(AtomicU64::new(base));
        let sampler = {
            let (stop, peak) = (Arc::clone(&stop), Arc::clone(&peak));
            std::thread::spawn(move || {
                let device = mlx_native::MlxDevice::new().unwrap();
                while !stop.load(Ordering::Relaxed) {
                    let now = device.metal_device().current_allocated_size() as u64;
                    peak.fetch_max(now, Ordering::Relaxed);
                    std::thread::sleep(Duration::from_micros(250));
                }
            })
        };
        let begin = Instant::now();
        let images: Vec<LaneImage> = if lanes == 1 {
            let first = resume(&mut guard, 0, &prompts[0], depth);
            let logits = guard.model.weights.logits_view().unwrap().to_vec();
            vec![lane_image(&guard, 0, depth + rows, first, &logits)]
        } else {
            let seqs: Vec<(Vec<u32>, SlotId, usize)> = (0..lanes)
                .map(|slot| (prompts[slot][depth..].to_vec(), SlotId(slot as u32), depth))
                .collect();
            clear_gemma4_self_mounts(guard.model);
            let reserve = gemma4_batch_decode_reserve(
                (0..lanes).map(|slot| (prompts[slot].len(), MAX_DECODE)),
            );
            let output = guard.model.weights.forward_prefill_batched_multi_seq_live(
                &seqs,
                guard.hybrid.as_deref().unwrap(),
                reserve,
                &mut guard.model.ctx,
            );
            clear_gemma4_self_mounts(guard.model);
            let output = output.unwrap();
            let commits: Vec<_> = (0..lanes)
                .map(|slot| (SlotId(slot as u32), depth, depth + rows))
                .collect();
            commit_gemma4_slot_cursors(&mut guard, &commits).unwrap();
            output
                .first_tokens
                .into_iter()
                .zip(output.logits)
                .enumerate()
                .map(|(slot, (first, logits))| {
                    lane_image(&guard, slot, depth + rows, first, &logits)
                })
                .collect()
        };
        let wall = begin.elapsed();
        let stats = last_forward_scratch_stats();
        stop.store(true, Ordering::Relaxed);
        sampler.join().unwrap();
        set_test_tensor_scratch_limits(None);
        let peak_extra = peak.load(Ordering::Relaxed).saturating_sub(base);
        assert!(
            images.iter().all(|image| image
                .logits
                .iter()
                .all(|bits| f32::from_bits(*bits).is_finite())),
            "{label}: non-finite logits at history {depth}"
        );
        eprintln!(
            "GEMMA_WARM_DEPTH ARM lanes={lanes} history={depth} rows={rows} arm={label} wall_ms={:.0} \
             peak_extra_mib={} accounted_global_scratch_mib={} waits={} first_tokens={:?}",
            wall.as_secs_f64() * 1e3,
            peak_extra / MIB,
            stats.peak_bytes / MIB,
            stats.waits,
            images.iter().map(|image| image.first_token).collect::<Vec<_>>()
        );
        arms.push(Arm {
            label,
            wall,
            peak_extra,
            stats,
            lanes: images,
        });
    }

    let arm = |label: &str| arms.iter().find(|arm| arm.label == label);
    if let Some(bounded) = arm("bounded") {
        // Envelope: one sequence's dense K/V of one global layer (or the
        // budget when that is smaller) plus one mask.
        let pair = GLOBAL_DENSE_BYTES_PER_KEY * (depth + rows + 64) as u64;
        let envelope = pair.max(TENSOR_GLOBAL_SCRATCH_BUDGET_BYTES) + TENSOR_GLOBAL_MASK_BYTES_MAX;
        assert!(
            bounded.stats.peak_bytes <= envelope,
            "accounted global scratch {} MiB exceeds the envelope {} MiB",
            bounded.stats.peak_bytes / MIB,
            envelope / MIB
        );
        if let Some(vec) = arm("vec") {
            // Measured: the tensor default adds at most the envelope (and
            // its sliding-layer stages, bounded by the suffix rows) to the
            // kill-switch route's transient allocation.
            let sliding_allowance = 1_024 * MIB;
            assert!(
                bounded.peak_extra <= vec.peak_extra + envelope + sliding_allowance,
                "tensor default peak {} MiB against vec-hybrid {} MiB and envelope {} MiB",
                bounded.peak_extra / MIB,
                vec.peak_extra / MIB,
                envelope / MIB
            );
            eprintln!(
                "GEMMA_WARM_DEPTH speedup bounded-vs-vec {:.2}x",
                vec.wall.as_secs_f64() / bounded.wall.as_secs_f64()
            );
        }
        if let Some(unbounded) = arm("unbounded") {
            for (slot, (got, want)) in bounded.lanes.iter().zip(&unbounded.lanes).enumerate() {
                assert_lane_equal(&format!("bounded vs unbounded lane {slot}"), got, want);
            }
            eprintln!(
                "GEMMA_WARM_DEPTH bounded == unbounded bitwise lanes={lanes} history={depth} \
                 (peak {} MiB against {} MiB)",
                bounded.peak_extra / MIB,
                unbounded.peak_extra / MIB
            );
        }
    }
}
