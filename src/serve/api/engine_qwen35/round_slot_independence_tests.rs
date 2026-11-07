//! ADR-063 B gate 2 (official artifact): a speculative round on one slot is
//! independent of the other slots (Q1 §3.1 Class A).
//!
//! Each gate loads the official artifact once with speculation `auto` (the
//! default repeatable route) and serves the same target request in a fresh
//! four-slot cache. The reference runs it in slot 2 with the other slots
//! untouched; each variant changes one thing:
//!
//! 1. **Repeat.** The reference again: same slot, empty peers. This catches
//!    run-to-run nondeterminism, e.g. the missing MTP gate-multiply barrier
//!    that changed K3 drafts between identical rounds.
//! 2. **Prefilled.** After the target's prefill, each peer slot is prefilled
//!    with its own prompt.
//! 3. **Mid-decode.** As 2, then each peer runs a few decode ticks (its
//!    ordinary warm-up decision and its own speculative rounds).
//! 4. **Interleaved.** As 3, and every live peer also ticks once in each
//!    scheduler turn, in slot order around the target.
//! 5. **Target slot.** Empty peers, the target in slot 0, 1 or 3: a server
//!    may place the same request in any slot.
//!
//! The target is driven like a server lane: bounded prefill through
//! `Qwen35PrefillState`, then `Qwen35DecodeState::decode_tick` until it
//! finishes (`TARGET_MAX_TOKENS`). Two targets run:
//! - **copy:** a verbatim copy of a code block. The first speculative tick
//!   (assistant-header suffix) finds no history match and runs K3; once the
//!   copy starts, history lookup finds drafts. The reference must run both.
//! - **k3:** free generation, where history rarely matches and K3 chains run
//!   for dozens of rounds.
//!
//! Each tick's route is read from the process-global speculation counters
//! around the target's own tick (run with `--test-threads=1`). The repeatable
//! route's cost gates are modelled, so a gate disable is part of the compared
//! route sequence; the reference route is printed one letter per tick for
//! comparison across processes.
//!
//! Compared bitwise against the reference:
//! - the target's prefill logits;
//! - per tick: the route, committed tokens, verified queue, target and MTP
//!   cursors, and the MTP verifier hidden row;
//! - after the rounds: per-layer cursors, DeltaNet parity and conv/recurrent
//!   current and scratch regions, the valid rows of every full-attention
//!   layer's KV codes/scales (TQ) or F32 K/V, and the valid rows of the MTP
//!   cache, all for the target slot;
//! - the next-token logits of one ordinary probe forward over the final
//!   target state.
//!
//! Rows at or above a cursor are not observable and are not compared.
//!
//! ```text
//! # The Qwen3.8 padded route comes from the model's own routing policy
//! # (artifact path names qwen3.8/qwen38; no HF2Q_DECODE_MVN/MV_EXT override).
//! HF2Q_QWEN38_GGUF=/abs/Qwen3.8-27B-vanilla.gguf \
//!   cargo test --release --bin hf2q qwen38_k3_and_history_rounds_are_slot_independent_bitwise \
//!   -- --ignored --exact --nocapture --test-threads=1
//! HF2Q_QWEN36_GGUF=/abs/qwen36.gguf \
//!   cargo test --release --bin hf2q qwen36_k3_and_history_rounds_are_slot_independent_bitwise \
//!   -- --ignored --exact --nocapture --test-threads=1
//! ```

use std::ops::Range;
use std::path::PathBuf;
use std::sync::atomic::Ordering;

use mlx_native::{DType, MlxBuffer};

use super::{Qwen35DecodeState, Qwen35LoadedModel, Qwen35PrefillAdvance, Qwen35PrefillState};
use crate::inference::models::qwen35::kv_cache::{FullAttnKvSlot, HybridKvCache};
use crate::serve::api::engine::{LoadOptions, LoadedModel, SamplingParams};
use crate::serve::api::engine_supervisor::EngineSupervisor;
use crate::serve::api::qwen35_speculation::{
    QwenSpeculationController, QwenSpeculationPolicy, TELEMETRY,
};
use crate::serve::multi_seq_kv::SlotId;

const SLOTS: u32 = 4;
/// The reference target slot; variants also place the target in the others.
pub(super) const REFERENCE_SLOT: SlotId = SlotId(2);
const CAPACITY: u32 = 1024;
pub(super) const PREFILL_CHUNK: usize = 512;
pub(super) const TARGET_MAX_TOKENS: usize = 192;
pub(super) const PEER_MAX_TOKENS: usize = 48;
pub(super) const PEER_DECODE_STEPS: usize = 4;
/// Every tick commits one token, so a target that runs to its bound takes at
/// most this many ticks.
pub(super) const MAX_TARGET_TICKS: usize = TARGET_MAX_TOKENS + 1;

pub(super) const TARGET_TEXT: &str =
    "Copy this Python function exactly, as one code block with no other text.\n\n\
```python\n\
def merge_intervals(intervals):\n    \
ordered = sorted(intervals, key=lambda pair: pair[0])\n    \
merged = []\n    \
for start, end in ordered:\n        \
if merged and start <= merged[-1][1]:\n            \
merged[-1][1] = max(merged[-1][1], end)\n        \
else:\n            \
merged.append([start, end])\n    \
return merged\n\
```";
/// Free generation: history lookup rarely matches, so K3 rounds dominate.
pub(super) const K3_TARGET_TEXT: &str =
    "Explain step by step how inserting a key into a binary search tree works, \
then show a short Rust implementation of the insert function.";
pub(super) const PEER_TEXTS: [&str; 3] = [
    "List the first ten prime numbers, one per line.",
    "Explain in two sentences why the sky looks blue during the day.",
    "Write a haiku about a lighthouse in winter fog, then give it a title.",
];

/// What slots 0, 1 and 3 hold while the target runs its rounds.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Peers {
    Empty,
    Prefilled,
    MidDecode,
    Interleaved,
}

/// Route one target tick took.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Route {
    /// One target decision without speculation (cursor + 1).
    Ordinary,
    /// A queued verified token, no forward (cursor unchanged).
    Commit,
    K3,
    History,
}

/// The target's observable state after one tick.
pub(super) struct Tick {
    route: Route,
    generated: Vec<u32>,
    pending: Vec<u32>,
    pub(super) finished: bool,
    target_cursor: u32,
    mtp_cursor: u32,
    verifier_hidden: Vec<u32>,
}

/// Every observable byte of the target slot: per layer its cursor and the
/// valid K/V rows, the MTP cache likewise, and per DeltaNet layer its parity
/// and conv/recurrent current and scratch regions.
#[derive(PartialEq, Eq)]
pub(super) struct SlotState {
    full_attn: Vec<(u32, Vec<Vec<u8>>)>,
    mtp: (u32, Vec<Vec<u8>>),
    linear_attn: Vec<(bool, Vec<Vec<u8>>)>,
}

pub(super) struct TargetTrace {
    pub(super) prefill_logits: Vec<u32>,
    pub(super) ticks: Vec<Tick>,
    pub(super) state: SlotState,
    pub(super) probe_logits: Vec<u32>,
}

impl TargetTrace {
    pub(super) fn routes(&self) -> Vec<Route> {
        self.ticks.iter().map(|tick| tick.route).collect()
    }
}

pub(super) struct Peer {
    pub(super) slot: SlotId,
    pub(super) state: Qwen35DecodeState,
    pub(super) live: bool,
}

pub(super) fn user_turn(text: &str) -> String {
    format!("<|im_start|>user\n{text}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n")
}

pub(super) fn bits(values: &[f32]) -> Vec<u32> {
    values.iter().map(|value| value.to_bits()).collect()
}

/// Little-endian bytes of the logical elements `range` of a U8 or F32 buffer.
fn elements(buffer: &MlxBuffer, range: Range<usize>) -> Vec<u8> {
    match buffer.dtype() {
        DType::U8 => buffer
            .as_logical_slice::<u8>()
            .expect("read U8 cache buffer")[range]
            .to_vec(),
        DType::F32 => buffer
            .as_logical_slice::<f32>()
            .expect("read F32 cache buffer")[range]
            .iter()
            .flat_map(|value| value.to_bits().to_le_bytes())
            .collect(),
        other => panic!("unexpected cache dtype {other:?}"),
    }
}

/// `slot`'s region of a buffer whose outermost axis is the slot.
fn slot_region(buffer: &MlxBuffer, slot: SlotId) -> Vec<u8> {
    let count = buffer.element_count();
    assert_eq!(count % SLOTS as usize, 0, "buffer not split by slot");
    let per_slot = count / SLOTS as usize;
    let start = slot.0 as usize * per_slot;
    elements(buffer, start..start + per_slot)
}

/// The observable rows `0..rows` of `slot` in a
/// `[n_seqs, n_kv_heads, capacity, inner]` K/V buffer.
fn valid_rows(buffer: &MlxBuffer, slot: SlotId, rows: u32) -> Vec<u8> {
    let shape = buffer.shape();
    assert!(
        shape.len() == 4 && shape[0] == SLOTS as usize,
        "K/V buffer shape {shape:?} is not [n_seqs, n_kv_heads, capacity, inner]"
    );
    let (heads, capacity, inner) = (shape[1], shape[2], shape[3]);
    let rows = rows as usize;
    assert!(rows <= capacity, "cursor {rows} beyond capacity {capacity}");
    (0..heads)
        .flat_map(|head| {
            let start = (slot.0 as usize * heads + head) * capacity * inner;
            elements(buffer, start..start + rows * inner)
        })
        .collect()
}

fn full_attn_rows(layer: &FullAttnKvSlot, slot: SlotId) -> (u32, Vec<Vec<u8>>) {
    let rows = layer.current_len[slot.0 as usize];
    let mut regions = Vec::new();
    if let Some(tq) = &layer.tq {
        for buffer in [&tq.k_packed, &tq.k_norms, &tq.v_packed, &tq.v_norms] {
            regions.push(valid_rows(buffer, slot, rows));
        }
    }
    for buffer in [&layer.k, &layer.v].into_iter().flatten() {
        regions.push(valid_rows(buffer, slot, rows));
    }
    assert!(
        !regions.is_empty(),
        "full-attention layer holds no K/V buffers"
    );
    (rows, regions)
}

pub(super) fn slot_state(kv: &HybridKvCache, slot: SlotId) -> SlotState {
    SlotState {
        full_attn: kv
            .full_attn
            .iter()
            .map(|layer| full_attn_rows(layer, slot))
            .collect(),
        mtp: full_attn_rows(kv.mtp_slot.as_ref().expect("MTP cache"), slot),
        linear_attn: kv
            .linear_attn
            .iter()
            .map(|layer| {
                (
                    layer.pp_flipped[slot.0 as usize],
                    [
                        &layer.conv_state,
                        &layer.conv_state_scratch,
                        &layer.recurrent,
                        &layer.recurrent_scratch,
                    ]
                    .into_iter()
                    .map(|buffer| slot_region(buffer, slot))
                    .collect(),
                )
            })
            .collect(),
    }
}

/// (target cursor, MTP cursor) of `slot`.
pub(super) fn cursors(kv: &HybridKvCache, slot: SlotId) -> (u32, u32) {
    (
        kv.sequence_len_for_slot(slot).expect("target cursor"),
        kv.mtp_slot.as_ref().expect("MTP cache").current_len[slot.0 as usize],
    )
}

/// (MTP rounds, history-lookup rounds) recorded so far in this process.
fn rounds_recorded() -> (u64, u64) {
    (
        TELEMETRY.mtp_proposals.load(Ordering::Relaxed),
        TELEMETRY.history_lookup_proposals.load(Ordering::Relaxed),
    )
}

pub(super) fn new_cache(qwen: &Qwen35LoadedModel) -> HybridKvCache {
    let kv = qwen
        .model
        .with_gpu_cache_mut(|device, _registry| {
            HybridKvCache::new_with_options(
                &qwen.model.cfg,
                device,
                CAPACITY,
                SLOTS,
                qwen.tq_kv_active,
            )
        })
        .expect("allocate four-slot cache");
    assert!(kv.mtp_slot.is_some(), "the cache needs the MTP block");
    kv
}

/// Bounded server prefill of `prompt` into `slot`; returns the seeded decode
/// state and the prompt-boundary logits.
pub(super) fn prefill(
    qwen: &mut Qwen35LoadedModel,
    kv: &mut HybridKvCache,
    slot: SlotId,
    prompt: &[u32],
    max_tokens: usize,
    supervisor: &EngineSupervisor,
) -> (Qwen35DecodeState, Vec<f32>) {
    let params = SamplingParams {
        max_tokens,
        ..SamplingParams::default()
    };
    let mut pending = Qwen35PrefillState::begin(
        prompt.to_vec(),
        params,
        None,
        kv,
        slot,
        0,
        None,
        None,
        None,
        qwen.hidden_size,
        None,
    )
    .unwrap_or_else(|error| panic!("slot {} prefill admission: {error:#}", slot.0));
    loop {
        match pending
            .advance(qwen, None, kv, PREFILL_CHUNK, supervisor)
            .unwrap_or_else(|error| panic!("slot {} prefill: {error:#}", slot.0))
        {
            Qwen35PrefillAdvance::Pending { state, .. } => pending = state,
            Qwen35PrefillAdvance::Ready {
                state,
                prefill_logits,
                ..
            } => {
                assert!(
                    !state.finished_at_seed(),
                    "slot {} finished at its seed token",
                    slot.0
                );
                return (state, prefill_logits);
            }
        }
    }
}

pub(super) fn tick_peer(
    qwen: &mut Qwen35LoadedModel,
    kv: &mut HybridKvCache,
    peer: &mut Peer,
    supervisor: &EngineSupervisor,
) {
    if !peer.live {
        return;
    }
    let outcome = peer
        .state
        .decode_tick(qwen, kv, supervisor)
        .unwrap_or_else(|error| panic!("peer slot {} tick: {error:#}", peer.slot.0));
    if peer.state.complete_tick(outcome).finished {
        peer.live = false;
    }
}

pub(super) fn tick_target(
    qwen: &mut Qwen35LoadedModel,
    kv: &mut HybridKvCache,
    slot: SlotId,
    state: &mut Qwen35DecodeState,
    supervisor: &EngineSupervisor,
) -> Tick {
    let (mtp_before, history_before) = rounds_recorded();
    let (cursor_before, _) = cursors(kv, slot);
    let outcome = state
        .decode_tick(qwen, kv, supervisor)
        .unwrap_or_else(|error| panic!("target tick: {error:#}"));
    let finished = state.complete_tick(outcome).finished;
    let (mtp_after, history_after) = rounds_recorded();
    let (target_cursor, mtp_cursor) = cursors(kv, slot);
    let route = match (mtp_after - mtp_before, history_after - history_before) {
        (0, 0) if target_cursor == cursor_before => Route::Commit,
        (0, 0) => Route::Ordinary,
        (1, 0) => Route::K3,
        (0, 1) => Route::History,
        rounds => panic!("one target tick recorded (MTP, history) rounds {rounds:?}"),
    };
    Tick {
        route,
        generated: state.generated_tokens.clone(),
        pending: state.pending_speculation_output.iter().copied().collect(),
        finished,
        target_cursor,
        mtp_cursor,
        verifier_hidden: state.mtp.as_ref().map_or_else(Vec::new, |mtp| {
            bits(
                mtp.verifier_hidden
                    .as_logical_slice::<f32>()
                    .expect("MTP verifier hidden"),
            )
        }),
    }
}

/// Serve the target request in `target_slot` of a fresh cache with `peers`
/// in the other slots, until it finishes.
pub(super) fn run_variant(
    qwen: &mut Qwen35LoadedModel,
    target_prompt: &[u32],
    peer_prompts: &[Vec<u32>; 3],
    target_slot: SlotId,
    peers: Peers,
) -> TargetTrace {
    let label = format!("{peers:?} target slot {}", target_slot.0);
    let supervisor = EngineSupervisor::new();
    let mut kv = new_cache(qwen);
    let (mut target, prefill_logits) = prefill(
        qwen,
        &mut kv,
        target_slot,
        target_prompt,
        TARGET_MAX_TOKENS,
        &supervisor,
    );
    assert!(target.mtp.is_some(), "target prefill must seed MTP state");
    assert!(
        target.history_lookup.is_some(),
        "target must be eligible for history lookup"
    );
    kv.validate_speculative_cursors_for_slot(target_slot, target_prompt.len())
        .expect("target and MTP cursors at the prompt boundary");

    let mut lanes: Vec<Peer> = Vec::new();
    if peers != Peers::Empty {
        let peer_slots = (0..SLOTS).map(SlotId).filter(|&slot| slot != target_slot);
        for (slot, prompt) in peer_slots.zip(peer_prompts) {
            let (state, _) = prefill(qwen, &mut kv, slot, prompt, PEER_MAX_TOKENS, &supervisor);
            lanes.push(Peer {
                slot,
                state,
                live: true,
            });
        }
    }
    if matches!(peers, Peers::MidDecode | Peers::Interleaved) {
        for _ in 0..PEER_DECODE_STEPS {
            for lane in &mut lanes {
                tick_peer(qwen, &mut kv, lane, &supervisor);
            }
        }
        for lane in &lanes {
            assert!(
                lane.state.generated_tokens.len() > 1,
                "{label}: peer slot {} did not advance",
                lane.slot.0
            );
        }
    }

    let mut ticks: Vec<Tick> = Vec::new();
    while !ticks.last().is_some_and(|tick: &Tick| tick.finished) {
        assert!(
            ticks.len() < MAX_TARGET_TICKS,
            "{label}: target did not finish within {MAX_TARGET_TICKS} ticks"
        );
        // One scheduler turn, lanes in slot order.
        for slot in (0..SLOTS).map(SlotId) {
            if slot == target_slot {
                ticks.push(tick_target(qwen, &mut kv, slot, &mut target, &supervisor));
            } else if peers == Peers::Interleaved {
                if let Some(lane) = lanes.iter_mut().find(|lane| lane.slot == slot) {
                    tick_peer(qwen, &mut kv, lane, &supervisor);
                }
            }
        }
    }
    assert!(supervisor.is_healthy(), "{label}: GPU supervision failed");

    let state = slot_state(&kv, target_slot);
    let next = *target
        .generated_tokens
        .last()
        .expect("target committed tokens");
    let position = target_prompt.len() + target.generated_tokens.len() - 1;
    // A request that finishes inside a verify round leaves its cursor past
    // the newest committed token (the slot is released, never rolled back);
    // the probe then has no defined input and is skipped. The per-tick
    // cursors already compare that end state.
    let probe = if cursors(&kv, target_slot).0 as usize == position {
        qwen.model
            .forward_gpu_last_logits(&[next], &[position as i32; 4], &mut kv, target_slot)
            .unwrap_or_else(|error| panic!("{label}: probe forward: {error:#}"))
    } else {
        Vec::new()
    };
    TargetTrace {
        prefill_logits: bits(&prefill_logits),
        ticks,
        state,
        probe_logits: bits(&probe),
    }
}

pub(super) fn assert_same(reference: &TargetTrace, variant: &TargetTrace, peers: &str) {
    assert!(
        variant.prefill_logits == reference.prefill_logits,
        "{peers:?}: target prefill logits differ"
    );
    // Per tick, inputs of the next decision first (hidden row, cursors),
    // then the route, so the first report names the earliest difference.
    for (index, (got, want)) in variant.ticks.iter().zip(&reference.ticks).enumerate() {
        let at = format!("{peers:?} tick {index} ({:?})", want.route);
        assert_eq!(got.route, want.route, "{at}: route differs");
        assert_eq!(
            got.generated, want.generated,
            "{at}: committed tokens differ"
        );
        assert_eq!(got.pending, want.pending, "{at}: verified queue differs");
        assert_eq!(got.finished, want.finished, "{at}: finish differs");
        assert_eq!(
            (got.target_cursor, got.mtp_cursor),
            (want.target_cursor, want.mtp_cursor),
            "{at}: (target, MTP) cursors differ"
        );
        if got.verifier_hidden != want.verifier_hidden {
            let max_abs = got
                .verifier_hidden
                .iter()
                .zip(&want.verifier_hidden)
                .map(|(&a, &b)| (f32::from_bits(a) - f32::from_bits(b)).abs())
                .fold(0.0f32, f32::max);
            let count = got
                .verifier_hidden
                .iter()
                .zip(&want.verifier_hidden)
                .filter(|(a, b)| a != b)
                .count();
            panic!(
                "{at}: MTP verifier hidden row differs in {count} elements, max |diff| {max_abs:e}"
            );
        }
    }
    assert_eq!(
        variant.routes(),
        reference.routes(),
        "{peers:?}: target route sequence length differs"
    );

    let (state, expected) = (&variant.state, &reference.state);
    assert_eq!(state.full_attn.len(), expected.full_attn.len());
    for (layer, (got, want)) in state.full_attn.iter().zip(&expected.full_attn).enumerate() {
        assert_eq!(
            got.0, want.0,
            "{peers:?}: full-attention layer {layer} cursor differs"
        );
        assert!(
            got.1 == want.1,
            "{peers:?}: full-attention layer {layer} KV bytes differ"
        );
    }
    assert_eq!(
        state.mtp.0, expected.mtp.0,
        "{peers:?}: MTP cache cursor differs"
    );
    assert!(
        state.mtp.1 == expected.mtp.1,
        "{peers:?}: MTP cache KV bytes differ"
    );
    assert_eq!(state.linear_attn.len(), expected.linear_attn.len());
    const LINEAR: [&str; 4] = ["conv", "conv scratch", "recurrent", "recurrent scratch"];
    for (layer, (got, want)) in state
        .linear_attn
        .iter()
        .zip(&expected.linear_attn)
        .enumerate()
    {
        assert_eq!(
            got.0, want.0,
            "{peers:?}: DeltaNet layer {layer} parity differs"
        );
        for (name, (got_buffer, want_buffer)) in LINEAR.iter().zip(got.1.iter().zip(&want.1)) {
            assert!(
                got_buffer == want_buffer,
                "{peers:?}: DeltaNet layer {layer} {name} state differs"
            );
        }
    }
    assert!(
        variant.probe_logits == reference.probe_logits,
        "{peers:?}: post-round probe logits differ"
    );
}

/// Load the official artifact named by `env` for a bitwise gate: native MTP
/// head, no graft, speculation `auto` (the default repeatable route), GPU
/// cache primed.
pub(super) fn load_gate_artifact(env: &str) -> LoadedModel {
    let path = PathBuf::from(
        std::env::var_os(env).unwrap_or_else(|| panic!("this gate requires {env}")),
    );
    assert!(path.is_file(), "{env} is not a file: {}", path.display());
    let options = LoadOptions {
        model_path: path,
        tokenizer_path: None,
        config_path: None,
        dwq_overlay_path: None,
        kv_persist_dir: None,
        kv_persist_budget_bytes: 0,
        glp_path: None,
        glp_alpha: None,
        kv_graft_path: None,
        lcp_resume_budget: crate::serve::kv_persist::lcp_registry::LcpResumeBudget::DEFAULT,
    };
    let mut loaded = LoadedModel::load(&options).expect("load Qwen artifact");
    let LoadedModel::Qwen35(qwen) = &mut loaded else {
        panic!("{env} must name a Qwen3.5-family GGUF")
    };
    assert!(
        qwen.model.mtp.is_some(),
        "the gate needs the native MTP head"
    );
    assert!(qwen.model.kv_graft.is_none(), "no graft may be bound");
    qwen.speculation = QwenSpeculationController::new(QwenSpeculationPolicy::Auto);
    qwen.model
        .ensure_gpu_cache_primed()
        .expect("prime GPU cache before allocating caches");
    loaded
}

/// The padded Qwen3.8 gates run on the model's own frozen routing policy
/// (#729), never on process environment the test sets.
pub(super) fn assert_padded_policy(qwen: &Qwen35LoadedModel, env: &str) {
    let policy = qwen.model.ggml_routing_policy();
    assert!(
        !policy.dense_decode_mvn && policy.dense_decode_mv_ext,
        "the padded Qwen3.8 gate needs the model's serve policy (mvN off, mul_mv_ext on): \
         {env} must name a Qwen3.8 artifact (qwen3.8/qwen38, any case) and \
         HF2Q_DECODE_MVN/HF2Q_DECODE_MV_EXT must not override it"
    );
}

fn run_gate(env: &str, expect_padded: bool) {
    let _gpu = crate::inference::hf2q_gpu_test_lock();
    let mut loaded = load_gate_artifact(env);
    let LoadedModel::Qwen35(qwen) = &mut loaded else {
        unreachable!("load_gate_artifact returns a Qwen model")
    };
    if expect_padded {
        assert_padded_policy(qwen, env);
    }

    let tokenizer = qwen.tokenizer.clone();
    let encode = |text: &str| -> Vec<u32> {
        tokenizer
            .encode(user_turn(text).as_str(), false)
            .expect("tokenize gate prompt")
            .get_ids()
            .to_vec()
    };
    let peer_prompts = PEER_TEXTS.map(encode);

    // The copy target exercises history rounds, the free-generation target
    // long K3 chains (dozens of rounds, where serving first showed MTP
    // acceptance changing between processes).
    for (name, text, needs_history) in [("copy", TARGET_TEXT, true), ("k3", K3_TARGET_TEXT, false)]
    {
        let target_prompt = encode(text);
        let reference = run_variant(
            qwen,
            &target_prompt,
            &peer_prompts,
            REFERENCE_SLOT,
            Peers::Empty,
        );
        let routes = reference.routes();
        let k3_rounds = routes.iter().filter(|&&route| route == Route::K3).count();
        // One letter per tick, comparable across processes.
        let code: String = routes
            .iter()
            .map(|route| match route {
                Route::Ordinary => 'O',
                Route::Commit => 'c',
                Route::K3 => 'K',
                Route::History => 'H',
            })
            .collect();
        eprintln!(
            "[slot-independence] {env} {name}: {} target ticks, {k3_rounds} K3 rounds, routes {code}",
            routes.len()
        );
        assert!(k3_rounds > 0, "{name}: no K3 round ran: {routes:?}");
        assert!(
            !needs_history || routes.contains(&Route::History),
            "{name}: no history round ran: {routes:?}"
        );
        // `Empty` in the reference slot again is the in-process repeat.
        let variants = [
            Peers::Empty,
            Peers::Prefilled,
            Peers::MidDecode,
            Peers::Interleaved,
        ]
        .map(|peers| (REFERENCE_SLOT, peers))
        .into_iter()
        .chain([0, 1, 3].map(|slot| (SlotId(slot), Peers::Empty)));
        for (slot, peers) in variants {
            let variant = run_variant(qwen, &target_prompt, &peer_prompts, slot, peers);
            let label = format!("{name}: {peers:?} target slot {}", slot.0);
            assert_same(&reference, &variant, &label);
            eprintln!("[slot-independence] {env}: {label} bitwise equal");
        }
    }
}

/// Class A (Q1 §3.1) on the official Qwen3.8 artifact, padded `mul_mv_ext`
/// decode route: a request's K3 MTP and history-lookup rounds leave
/// bitwise-identical logits, hidden rows, committed tokens, cursors, KV
/// codes/scales, conv/recurrent state and MTP cache whatever slot it runs in
/// and whether the other slots are empty, prefilled, mid-decode, or decoding
/// in the same turns.
#[test]
#[ignore = "requires HF2Q_QWEN38_GGUF (official Qwen3.8 artifact with its MTP head) and Apple Metal"]
fn qwen38_k3_and_history_rounds_are_slot_independent_bitwise() {
    run_gate("HF2Q_QWEN38_GGUF", true);
}

/// Class A (Q1 §3.1) on the official Qwen3.6 artifact: a request's K3 MTP and
/// history-lookup rounds leave bitwise-identical logits, hidden rows,
/// committed tokens, cursors, KV codes/scales, conv/recurrent state and MTP
/// cache whatever slot it runs in and whether the other slots are empty,
/// prefilled, mid-decode, or decoding in the same turns.
#[test]
#[ignore = "requires HF2Q_QWEN36_GGUF (official Qwen3.6 artifact with its MTP head) and Apple Metal"]
fn qwen36_k3_and_history_rounds_are_slot_independent_bitwise() {
    run_gate("HF2Q_QWEN36_GGUF", false);
}
