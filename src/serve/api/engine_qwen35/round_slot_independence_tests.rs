//! ADR-063 B gate 2 (official artifact): a speculative round on one slot is
//! independent of the other slots (Q1 §3.1 Class A).
//!
//! Each gate loads the official artifact once with speculation `auto` (the
//! default repeatable route) and serves the same target request in slot 2 of
//! a fresh four-slot cache, once per peer state of slots 0, 1 and 3:
//!
//! 1. **Empty.** The peers are never touched (the reference).
//! 2. **Prefilled.** After the target's prefill, each peer is prefilled with
//!    its own prompt.
//! 3. **Mid-decode.** As 2, then each peer runs a few decode ticks (its
//!    ordinary warm-up decision and its own speculative rounds).
//! 4. **Interleaved.** As 3, and every live peer also ticks once in each
//!    scheduler turn, in slot order around the target.
//!
//! The target is driven like a server lane: bounded prefill through
//! `Qwen35PrefillState`, then `Qwen35DecodeState::decode_tick` until it has
//! run at least one K3 MTP round and one history-lookup round, then until its
//! verified queue drains. Its prompt asks for a verbatim copy of a code
//! block: the first speculative tick (assistant-header suffix) finds no
//! history match and runs K3, and once the copy starts history lookup finds
//! drafts. Each tick's route is read from the process-global speculation
//! counters around the target's own tick (run with `--test-threads=1`).
//! Every proposer stays below eight rounds, so neither cost gate can complete
//! the two evaluation windows a disable needs.
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
//! HF2Q_DECODE_MVN=0 HF2Q_DECODE_MV_EXT=1 \
//! HF2Q_QWEN38_GGUF=/abs/Qwen3.8-27B-vanilla.gguf \
//!   cargo test --release --lib qwen38_k3_and_history_rounds_are_slot_independent_bitwise \
//!   -- --ignored --exact --nocapture --test-threads=1
//! HF2Q_QWEN36_GGUF=/abs/qwen36.gguf \
//!   cargo test --release --lib qwen36_k3_and_history_rounds_are_slot_independent_bitwise \
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
const TARGET: SlotId = SlotId(2);
const PEERS: [SlotId; 3] = [SlotId(0), SlotId(1), SlotId(3)];
const CAPACITY: u32 = 1024;
const PREFILL_CHUNK: usize = 512;
const TARGET_MAX_TOKENS: usize = 96;
const PEER_MAX_TOKENS: usize = 48;
const PEER_DECODE_STEPS: usize = 4;
const MAX_TARGET_TICKS: usize = 64;
/// A cost gate evaluates every four rounds and disables a proposer only
/// after two unprofitable windows, so at most seven rounds per proposer keep
/// the route sequence independent of the gate.
const MAX_ROUNDS_PER_PROPOSER: usize = 7;

const TARGET_TEXT: &str =
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
const PEER_TEXTS: [&str; 3] = [
    "List the first ten prime numbers, one per line.",
    "Explain in two sentences why the sky looks blue during the day.",
    "Write a haiku about a lighthouse in winter fog, then give it a title.",
];

/// What slots 0, 1 and 3 hold while the target runs its rounds.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Peers {
    Empty,
    Prefilled,
    MidDecode,
    Interleaved,
}

/// Route one target tick took.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Route {
    /// One target decision without speculation (cursor + 1).
    Ordinary,
    /// A queued verified token, no forward (cursor unchanged).
    Commit,
    K3,
    History,
}

/// The target's observable state after one tick.
struct Tick {
    route: Route,
    generated: Vec<u32>,
    pending: Vec<u32>,
    finished: bool,
    target_cursor: u32,
    mtp_cursor: u32,
    verifier_hidden: Vec<u32>,
}

/// Every observable byte of the target slot: per layer its cursor and the
/// valid K/V rows, the MTP cache likewise, and per DeltaNet layer its parity
/// and conv/recurrent current and scratch regions.
#[derive(PartialEq, Eq)]
struct SlotState {
    full_attn: Vec<(u32, Vec<Vec<u8>>)>,
    mtp: (u32, Vec<Vec<u8>>),
    linear_attn: Vec<(bool, Vec<Vec<u8>>)>,
}

struct TargetTrace {
    prefill_logits: Vec<u32>,
    ticks: Vec<Tick>,
    state: SlotState,
    probe_logits: Vec<u32>,
}

impl TargetTrace {
    fn routes(&self) -> Vec<Route> {
        self.ticks.iter().map(|tick| tick.route).collect()
    }
}

struct Peer {
    slot: SlotId,
    state: Qwen35DecodeState,
    live: bool,
}

fn user_turn(text: &str) -> String {
    format!("<|im_start|>user\n{text}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n")
}

fn bits(values: &[f32]) -> Vec<u32> {
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

fn slot_state(kv: &HybridKvCache, slot: SlotId) -> SlotState {
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
fn cursors(kv: &HybridKvCache, slot: SlotId) -> (u32, u32) {
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

fn new_cache(qwen: &Qwen35LoadedModel) -> HybridKvCache {
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
fn prefill(
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

fn tick_peer(
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

fn tick_target(
    qwen: &mut Qwen35LoadedModel,
    kv: &mut HybridKvCache,
    state: &mut Qwen35DecodeState,
    supervisor: &EngineSupervisor,
) -> Tick {
    let (mtp_before, history_before) = rounds_recorded();
    let (cursor_before, _) = cursors(kv, TARGET);
    let outcome = state
        .decode_tick(qwen, kv, supervisor)
        .unwrap_or_else(|error| panic!("target tick: {error:#}"));
    let finished = state.complete_tick(outcome).finished;
    let (mtp_after, history_after) = rounds_recorded();
    let (target_cursor, mtp_cursor) = cursors(kv, TARGET);
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

/// Serve the target request in a fresh cache with `peers` in the other
/// slots, until one K3 round and one history round ran and the verified
/// queue drained.
fn run_variant(
    qwen: &mut Qwen35LoadedModel,
    target_prompt: &[u32],
    peer_prompts: &[Vec<u32>; 3],
    peers: Peers,
) -> TargetTrace {
    let supervisor = EngineSupervisor::new();
    let mut kv = new_cache(qwen);
    let (mut target, prefill_logits) = prefill(
        qwen,
        &mut kv,
        TARGET,
        target_prompt,
        TARGET_MAX_TOKENS,
        &supervisor,
    );
    assert!(target.mtp.is_some(), "target prefill must seed MTP state");
    assert!(
        target.history_lookup.is_some(),
        "target must be eligible for history lookup"
    );
    kv.validate_speculative_cursors_for_slot(TARGET, target_prompt.len())
        .expect("target and MTP cursors at the prompt boundary");

    let mut lanes: Vec<Peer> = Vec::new();
    if peers != Peers::Empty {
        for (&slot, prompt) in PEERS.iter().zip(peer_prompts) {
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
                "{peers:?}: peer slot {} did not advance",
                lane.slot.0
            );
        }
    }

    let mut ticks: Vec<Tick> = Vec::new();
    let (mut k3_rounds, mut history_rounds) = (0usize, 0usize);
    loop {
        let rounds_ran = k3_rounds > 0 && history_rounds > 0;
        if ticks.last().is_some_and(|tick| tick.finished) {
            assert!(
                rounds_ran,
                "{peers:?}: target finished before a K3 and a history round ran: {:?}",
                ticks.iter().map(|tick| tick.route).collect::<Vec<_>>()
            );
            break;
        }
        if rounds_ran && target.pending_speculation_output.is_empty() {
            break;
        }
        assert!(
            ticks.len() < MAX_TARGET_TICKS
                && k3_rounds <= MAX_ROUNDS_PER_PROPOSER
                && history_rounds <= MAX_ROUNDS_PER_PROPOSER,
            "{peers:?}: no K3 and history round within the bounds: {:?}",
            ticks.iter().map(|tick| tick.route).collect::<Vec<_>>()
        );
        // One scheduler turn, lanes in slot order.
        for slot in (0..SLOTS).map(SlotId) {
            if slot == TARGET {
                let tick = tick_target(qwen, &mut kv, &mut target, &supervisor);
                match tick.route {
                    Route::K3 => k3_rounds += 1,
                    Route::History => history_rounds += 1,
                    Route::Ordinary | Route::Commit => {}
                }
                ticks.push(tick);
            } else if peers == Peers::Interleaved {
                if let Some(lane) = lanes.iter_mut().find(|lane| lane.slot == slot) {
                    tick_peer(qwen, &mut kv, lane, &supervisor);
                }
            }
        }
    }
    assert!(supervisor.is_healthy(), "{peers:?}: GPU supervision failed");

    let state = slot_state(&kv, TARGET);
    let next = *target
        .generated_tokens
        .last()
        .expect("target committed tokens");
    let position = target_prompt.len() + target.generated_tokens.len() - 1;
    assert_eq!(
        cursors(&kv, TARGET).0 as usize,
        position,
        "{peers:?}: the newest committed token must be the next target input"
    );
    let probe = qwen
        .model
        .forward_gpu_last_logits(&[next], &[position as i32; 4], &mut kv, TARGET)
        .unwrap_or_else(|error| panic!("{peers:?}: probe forward: {error:#}"));
    TargetTrace {
        prefill_logits: bits(&prefill_logits),
        ticks,
        state,
        probe_logits: bits(&probe),
    }
}

fn assert_same(reference: &TargetTrace, variant: &TargetTrace, peers: Peers) {
    assert!(
        variant.prefill_logits == reference.prefill_logits,
        "{peers:?}: target prefill logits differ"
    );
    assert_eq!(
        variant.routes(),
        reference.routes(),
        "{peers:?}: target route sequence differs"
    );
    for (index, (got, want)) in variant.ticks.iter().zip(&reference.ticks).enumerate() {
        let at = format!("{peers:?} tick {index} ({:?})", want.route);
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
        assert!(
            got.verifier_hidden == want.verifier_hidden,
            "{at}: MTP verifier hidden row differs"
        );
    }

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

fn run_gate(env: &str) {
    let _gpu = crate::inference::hf2q_gpu_test_lock();
    let path = PathBuf::from(
        std::env::var_os(env)
            .unwrap_or_else(|| panic!("the slot-independence gate requires {env}")),
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

    let tokenizer = qwen.tokenizer.clone();
    let encode = |text: &str| -> Vec<u32> {
        tokenizer
            .encode(user_turn(text).as_str(), false)
            .expect("tokenize gate prompt")
            .get_ids()
            .to_vec()
    };
    let target_prompt = encode(TARGET_TEXT);
    let peer_prompts = PEER_TEXTS.map(encode);

    let reference = run_variant(qwen, &target_prompt, &peer_prompts, Peers::Empty);
    let routes = reference.routes();
    eprintln!("[slot-independence] {env}: target routes {routes:?}");
    assert!(routes.contains(&Route::K3), "no K3 round ran: {routes:?}");
    assert!(
        routes.contains(&Route::History),
        "no history round ran: {routes:?}"
    );
    for peers in [Peers::Prefilled, Peers::MidDecode, Peers::Interleaved] {
        let variant = run_variant(qwen, &target_prompt, &peer_prompts, peers);
        assert_same(&reference, &variant, peers);
    }
}

/// Class A (Q1 §3.1) on the official Qwen3.8 artifact, padded `mul_mv_ext`
/// decode route: a K3 MTP round and a history-lookup round on slot 2 leave
/// bitwise-identical logits, hidden rows, committed tokens, cursors, KV
/// codes/scales, conv/recurrent state and MTP cache whether the other three
/// slots are empty, prefilled, mid-decode, or decoding in the same turns.
#[test]
#[ignore = "requires HF2Q_QWEN38_GGUF (official Qwen3.8 artifact with its MTP head), HF2Q_DECODE_MVN=0, HF2Q_DECODE_MV_EXT=1 and Apple Metal"]
fn qwen38_k3_and_history_rounds_are_slot_independent_bitwise() {
    for (name, value) in [("HF2Q_DECODE_MVN", "0"), ("HF2Q_DECODE_MV_EXT", "1")] {
        assert_eq!(
            std::env::var(name).as_deref(),
            Ok(value),
            "the padded Qwen3.8 gate requires externally-set {name}={value}"
        );
    }
    run_gate("HF2Q_QWEN38_GGUF");
}

/// Class A (Q1 §3.1) on the official Qwen3.6 artifact: a K3 MTP round and a
/// history-lookup round on slot 2 leave bitwise-identical logits, hidden
/// rows, committed tokens, cursors, KV codes/scales, conv/recurrent state and
/// MTP cache whether the other three slots are empty, prefilled, mid-decode,
/// or decoding in the same turns.
#[test]
#[ignore = "requires HF2Q_QWEN36_GGUF (official Qwen3.6 artifact with its MTP head) and Apple Metal"]
fn qwen36_k3_and_history_rounds_are_slot_independent_bitwise() {
    run_gate("HF2Q_QWEN36_GGUF");
}
