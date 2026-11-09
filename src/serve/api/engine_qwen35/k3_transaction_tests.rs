//! ADR-065 J1 commit-transaction tests (T-J1-3, T-J1-4, T-J1-5).
//!
//! A tiny four-slot hybrid cache with an MTP block stands in for the state a
//! real verify forward leaves behind: every buffer holds a distinct
//! deterministic pattern, the committing slot has four verifier rows on both
//! cursors, its DeltaNet parity is flipped and capture is armed. Plans come
//! from the production accept walk (`plan_qwen35_verified_block`) driven by
//! scripted canonical decisions; the decode state is the production
//! `Qwen35DecodeState` over a weightless loaded model.

use std::time::Duration;

use mlx_native::{DType, MlxBuffer, MlxDevice};

use super::super::speculation_qualification_tests::loaded_model;
use super::super::{
    plan_qwen35_verified_block, Qwen35CanonicalDecision, Qwen35DecodeState, Qwen35LoadedModel,
    Qwen35VerifiedBlockPlan,
};
use super::{K3Draft, K3Entry, K3LaRestore, K3Plan, K3PlanTiming, K3_VERIFY_ROWS};
use crate::inference::models::qwen35::kv_cache::{HybridKvCache, LaRestoreRoute};
use crate::inference::models::qwen35::Qwen35Config;
use crate::serve::api::engine::SamplingParams;
use crate::serve::api::engine_supervisor::EngineSupervisor;
use crate::serve::api::qwen35_speculation::QwenSpeculationPolicy;
use crate::serve::multi_seq_kv::{MultiSeqKvCache, SlotId};

const N_SEQS: u32 = 4;
const SLOT: SlotId = SlotId(2);
const PEERS: [(SlotId, u32); 3] = [(SlotId(0), 3), (SlotId(1), 11), (SlotId(3), 5)];
const PROMPT: [u32; 6] = [3, 9, 27, 17, 5, 41];
/// Entry cursor of the committing slot: prompt plus one ordinary decision.
const PRIOR: u32 = PROMPT.len() as u32 + 1;
const SEED_TOKEN: u32 = 10;
const DRAFTS: [u32; 3] = [11, 12, 13];
const MISS: u32 = 20;
const EOS: u32 = 63;

/// Canonical decisions per verifier row for the shapes of plan J1 commits.
const FULL: &[(u32, bool)] = &[(11, false), (12, false), (13, false), (14, false)];
const PARTIAL: &[(u32, bool)] = &[(11, false), (MISS, false)];
const ZERO: &[(u32, bool)] = &[(MISS, false)];
const TERMINAL_MATCH_AT_1: &[(u32, bool)] = &[(11, false), (12, true)];
const TERMINAL_MISS_AT_2: &[(u32, bool)] = &[(11, false), (12, false), (EOS, true)];
const TERMINAL_BONUS: &[(u32, bool)] = &[(11, false), (12, false), (13, false), (EOS, true)];

fn cache_cfg(qwen: &Qwen35LoadedModel) -> Qwen35Config {
    let mut cfg = qwen.model.cfg.clone();
    cfg.mtp_num_hidden_layers = 1;
    cfg
}

fn fill(buffer: &mut MlxBuffer, tag: u32) {
    for (index, value) in buffer
        .as_mut_slice::<f32>()
        .expect("CPU-visible test buffer")
        .iter_mut()
        .enumerate()
    {
        *value = tag as f32 * 1000.0 + (index % 997) as f32 * 0.25;
    }
}

fn hidden_rows(device: &MlxDevice, rows: usize, hidden: usize) -> MlxBuffer {
    let mut buffer = device
        .alloc_buffer(
            rows * hidden * std::mem::size_of::<f32>(),
            DType::F32,
            vec![rows, hidden],
        )
        .expect("verifier hidden rows");
    fill(&mut buffer, 900);
    buffer
}

/// The cache as a K3 round leaves it right before its commit.
fn verified_cache(cfg: &Qwen35Config, device: &MlxDevice) -> HybridKvCache {
    let mut kv = HybridKvCache::new(cfg, device, 64, N_SEQS).expect("four-slot cache with MTP");
    kv.ensure_la_capture(cfg, device, K3_VERIFY_ROWS as u32)
        .expect("arm recurrent capture");
    let mut tag = 1;
    let mut next = || {
        tag += 1;
        tag
    };
    for la in kv.linear_attn.iter_mut() {
        fill(&mut la.conv_state, next());
        fill(&mut la.conv_state_scratch, next());
        fill(&mut la.recurrent, next());
        fill(&mut la.recurrent_scratch, next());
        fill(
            la.capture_states.as_mut().expect("recurrent capture"),
            next(),
        );
        fill(
            la.conv_capture_states.as_mut().expect("conv capture"),
            next(),
        );
    }
    for full in kv.full_attn.iter_mut().chain(kv.mtp_slot.iter_mut()) {
        fill(full.k.as_mut().expect("F32 keys"), next());
        fill(full.v.as_mut().expect("F32 values"), next());
    }
    for (peer, cursor) in PEERS {
        kv.append_for_seq(peer, cursor).expect("peer cursor");
    }
    // Verify wrote four target rows, catch-up mirrored them into MTP, and
    // the verify forward flipped this slot's DeltaNet parity once. A peer
    // with flipped parity makes role confusion visible.
    kv.append_for_seq(SLOT, PRIOR + K3_VERIFY_ROWS as u32)
        .expect("verified cursors");
    for la in kv.linear_attn.iter_mut() {
        la.swap_for_slot(SLOT);
        la.swap_for_slot(SlotId(1));
    }
    kv
}

fn decode_state(qwen: &Qwen35LoadedModel, device: &MlxDevice) -> Qwen35DecodeState {
    let mut prefill_logits = vec![0.0f32; qwen.vocab_size];
    prefill_logits[SEED_TOKEN as usize] = 10.0;
    let mut verifier_hidden = device
        .alloc_buffer(
            qwen.hidden_size * std::mem::size_of::<f32>(),
            DType::F32,
            vec![1, qwen.hidden_size],
        )
        .expect("entry verifier hidden");
    fill(&mut verifier_hidden, 800);
    let params = SamplingParams {
        max_tokens: 64,
        ..SamplingParams::default()
    };
    Qwen35DecodeState::from_prefill_logits(
        qwen,
        PROMPT.to_vec(),
        params,
        None,
        SLOT,
        0,
        &prefill_logits,
        Duration::ZERO,
        PROMPT.len(),
        Some(verifier_hidden),
    )
    .expect("seeded decode state with MTP")
}

fn scripted_plan(decisions: &[(u32, bool)]) -> Qwen35VerifiedBlockPlan {
    plan_qwen35_verified_block(&DRAFTS, |row| {
        let (token, terminal) = decisions[row];
        Ok(Qwen35CanonicalDecision { token, terminal })
    })
    .expect("scripted canonical accept walk")
}

fn k3_plan(decisions: &[(u32, bool)], hidden: MlxBuffer) -> K3Plan {
    let draft = K3Draft {
        entry: K3Entry {
            next_token: SEED_TOKEN,
            next_pos: PRIOR as i32,
            prior_target_len: PRIOR,
            prior_mtp_len: PRIOR,
        },
        drafts: DRAFTS,
        verify_input: [SEED_TOKEN, DRAFTS[0], DRAFTS[1], DRAFTS[2]],
        verify_positions: Vec::new(),
    };
    let timing = K3PlanTiming {
        verify: Duration::ZERO,
        embed: Duration::ZERO,
        catchup: Duration::ZERO,
        semantic: Duration::ZERO,
    };
    K3Plan::new(draft, hidden, scripted_plan(decisions), timing)
}

fn bits(values: &[f32]) -> Vec<u32> {
    values.iter().map(|value| value.to_bits()).collect()
}

fn slot_region(buffer: &MlxBuffer, slot: SlotId) -> Vec<u32> {
    let values = buffer.as_slice::<f32>().expect("CPU-visible cache buffer");
    let per_seq = values.len() / N_SEQS as usize;
    let start = slot.0 as usize * per_seq;
    bits(&values[start..start + per_seq])
}

/// Everything a commit could touch for one slot: cursors, DeltaNet parity
/// and current/scratch state by role, and the slot's attention K/V regions.
#[derive(Debug, PartialEq, Eq)]
struct SlotDigest {
    target_cursors: Vec<u32>,
    mtp_cursor: u32,
    parity: Vec<bool>,
    linear: Vec<[Vec<u32>; 4]>,
    attention: Vec<[Vec<u32>; 2]>,
}

fn digest(kv: &HybridKvCache, slot: SlotId) -> SlotDigest {
    let index = slot.0 as usize;
    SlotDigest {
        target_cursors: kv.full_attn.iter().map(|f| f.current_len[index]).collect(),
        mtp_cursor: kv.mtp_slot.as_ref().expect("MTP slot").current_len[index],
        parity: kv
            .linear_attn
            .iter()
            .map(|la| la.pp_flipped[index])
            .collect(),
        linear: kv
            .linear_attn
            .iter()
            .map(|la| {
                let (conv, conv_scratch) = la.conv_bufs_for_slot(slot);
                let (recurrent, recurrent_scratch) = la.recurrent_bufs_for_slot(slot);
                [
                    slot_region(conv, slot),
                    slot_region(conv_scratch, slot),
                    slot_region(recurrent, slot),
                    slot_region(recurrent_scratch, slot),
                ]
            })
            .collect(),
        attention: kv
            .full_attn
            .iter()
            .chain(kv.mtp_slot.iter())
            .map(|full| {
                [
                    slot_region(full.k.as_ref().expect("keys"), slot),
                    slot_region(full.v.as_ref().expect("values"), slot),
                ]
            })
            .collect(),
    }
}

fn peer_digests(kv: &HybridKvCache) -> Vec<SlotDigest> {
    PEERS.iter().map(|&(peer, _)| digest(kv, peer)).collect()
}

/// The slot's current DeltaNet state that restoring `capture[slot, row]`
/// must produce for one layer: (conv, recurrent). Conv capture rows are
/// `[K-1, channels]`; the live conv state is `[channels, K-1]`.
fn capture_row(kv: &HybridKvCache, layer: usize, slot: SlotId, row: usize) -> [Vec<u32>; 2] {
    let la = &kv.linear_attn[layer];
    let read = |capture: &MlxBuffer, state: &MlxBuffer| {
        let per_seq = state.element_count() / N_SEQS as usize;
        let values = capture.as_slice::<f32>().expect("capture");
        let rows = values.len() / (per_seq * N_SEQS as usize);
        let start = slot.0 as usize * rows * per_seq + row * per_seq;
        values[start..start + per_seq].to_vec()
    };
    let conv_rows = read(
        la.conv_capture_states.as_ref().expect("conv capture"),
        &la.conv_state,
    );
    let (channels, k_minus1) = (la.conv_state.shape()[0], la.conv_state.shape()[1]);
    let mut conv = vec![0.0f32; conv_rows.len()];
    for k_i in 0..k_minus1 {
        for c in 0..channels {
            conv[c * k_minus1 + k_i] = conv_rows[k_i * channels + c];
        }
    }
    [
        bits(&conv),
        bits(&read(
            la.capture_states.as_ref().expect("recurrent capture"),
            &la.recurrent,
        )),
    ]
}

fn verifier_hidden_bits(state: &Qwen35DecodeState) -> Vec<u32> {
    bits(
        state
            .mtp
            .as_ref()
            .expect("MTP state")
            .verifier_hidden
            .as_logical_slice::<f32>()
            .expect("verifier hidden"),
    )
}

fn hidden_row_bits(hidden: &MlxBuffer, row: usize, width: usize) -> Vec<u32> {
    let values = hidden.as_logical_slice::<f32>().expect("hidden rows");
    bits(&values[row * width..(row + 1) * width])
}

fn assert_slot_reset(kv: &HybridKvCache) {
    let after = digest(kv, SLOT);
    assert!(after.target_cursors.iter().all(|&cursor| cursor == 0));
    assert_eq!(
        after.mtp_cursor, 0,
        "fail-closed reset rewinds the MTP cursor"
    );
    assert!(after.parity.iter().all(|&flipped| !flipped));
    assert!(
        after
            .linear
            .iter()
            .flatten()
            .flatten()
            .all(|&value| value == 0.0f32.to_bits()),
        "fail-closed reset zeroes this slot's DeltaNet state"
    );
}

/// T-J1-4: `k3_commit_cursors` commits `prior + valid_input_tokens` on both
/// cursors and owes a DeltaNet restore of the carry row exactly when the plan
/// is partial. It neither restores nor settles: capture stays armed for the
/// batched restore a joint step runs after every lane's cursors.
#[test]
fn k3_commit_cursors_commits_valid_prefix_and_owes_restore_only_when_partial() {
    let _gpu = crate::inference::hf2q_gpu_test_lock();
    let device = MlxDevice::new().expect("Metal device");
    let qwen = loaded_model(QwenSpeculationPolicy::Auto);
    let cfg = cache_cfg(&qwen);
    for (name, decisions, valid, restore) in [
        ("full", FULL, 4, None),
        ("terminal bonus", TERMINAL_BONUS, 4, None),
        ("partial", PARTIAL, 2, Some(1)),
        ("zero", ZERO, 1, Some(0)),
        ("terminal match at 1", TERMINAL_MATCH_AT_1, 2, Some(1)),
        ("terminal miss at 2", TERMINAL_MISS_AT_2, 3, Some(2)),
    ] {
        let mut kv = verified_cache(&cfg, &device);
        let state = decode_state(&qwen, &device);
        let plan = k3_plan(
            decisions,
            hidden_rows(&device, K3_VERIFY_ROWS, qwen.hidden_size),
        );
        assert_eq!(plan.plan.valid_input_tokens, valid, "{name}");
        let peers = peer_digests(&kv);
        let before = digest(&kv, SLOT);

        let owed = state
            .k3_commit_cursors(&mut kv, &plan)
            .unwrap_or_else(|error| panic!("{name}: {error:#}"));

        assert_eq!(
            owed,
            restore.map(|row| K3LaRestore {
                slot: SLOT,
                capture_row: row,
            }),
            "{name}"
        );
        if let Some(owed) = owed {
            assert_eq!(
                owed.capture_row as usize, plan.plan.carry_hidden_row,
                "{name}"
            );
        }
        let committed = PRIOR + valid as u32;
        assert_eq!(plan.committed_cursor, committed, "{name}");
        let after = digest(&kv, SLOT);
        assert!(
            after
                .target_cursors
                .iter()
                .all(|&cursor| cursor == committed),
            "{name}: target cursor"
        );
        assert_eq!(after.mtp_cursor, committed, "{name}: MTP cursor");
        assert_eq!(
            (after.parity, after.linear, after.attention),
            (before.parity, before.linear, before.attention),
            "{name}: cursor commit must not touch DeltaNet state or K/V bytes"
        );
        assert_eq!(
            kv.la_capture_active_tokens(),
            Some(K3_VERIFY_ROWS as u32),
            "{name}"
        );
        assert_eq!(peer_digests(&kv), peers, "{name}: peer slots");
    }
}

/// T-J1-3 (validation): a cursor-equality failure after truncation and
/// restore takes the fail-closed scope: capture cleared, this slot reset,
/// peers byte-identical, and no ledger state installed.
#[test]
fn solo_commit_validation_failure_clears_capture_resets_slot_and_installs_no_ledger() {
    let _gpu = crate::inference::hf2q_gpu_test_lock();
    let device = MlxDevice::new().expect("Metal device");
    let qwen = loaded_model(QwenSpeculationPolicy::Auto);
    let cfg = cache_cfg(&qwen);
    let mut kv = verified_cache(&cfg, &device);
    // The MTP catch-up never landed: the target holds four verifier rows,
    // MTP none, so truncation leaves MTP below the committed cursor.
    kv.truncate_mtp_to_for_slot(SLOT, PRIOR)
        .expect("MTP behind the target");
    let mut state = decode_state(&qwen, &device);
    let supervisor = EngineSupervisor::new();
    let generated = state.generated_tokens.clone();
    let hidden_before = verifier_hidden_bits(&state);
    let peers = peer_digests(&kv);
    let plan = k3_plan(
        PARTIAL,
        hidden_rows(&device, K3_VERIFY_ROWS, qwen.hidden_size),
    );

    let error = match state.k3_commit_solo(&qwen, &mut kv, &supervisor, plan) {
        Ok(_) => panic!("validation must fail with the MTP cursor behind"),
        Err(error) => format!("{error:#}"),
    };

    assert!(
        error.starts_with("Qwen SlotAware MTP K3 committed cursor equality"),
        "{error}"
    );
    assert!(!kv.la_capture_active(), "fail-closed scope clears capture");
    assert_slot_reset(&kv);
    assert_eq!(peer_digests(&kv), peers, "peer slots");
    assert!(state.pending_speculation_output.is_empty());
    assert!(!state.terminal_after_pending);
    assert_eq!(state.generated_tokens, generated);
    assert_eq!(verifier_hidden_bits(&state), hidden_before);
}

/// T-J1-3 (carry hidden): the carry row is read after the fail-closed scope,
/// so its failure only resets the slot. Capture was already cleared by the
/// settlement, the cursors were already committed, and no ledger state is
/// installed.
#[test]
fn solo_commit_carry_hidden_failure_resets_slot_after_settlement() {
    let _gpu = crate::inference::hf2q_gpu_test_lock();
    let device = MlxDevice::new().expect("Metal device");
    let qwen = loaded_model(QwenSpeculationPolicy::Auto);
    let cfg = cache_cfg(&qwen);
    let mut kv = verified_cache(&cfg, &device);
    let mut state = decode_state(&qwen, &device);
    let supervisor = EngineSupervisor::new();
    let generated = state.generated_tokens.clone();
    let hidden_before = verifier_hidden_bits(&state);
    let peers = peer_digests(&kv);
    // One hidden row: the partial plan's carry row 1 is out of range.
    let plan = k3_plan(PARTIAL, hidden_rows(&device, 1, qwen.hidden_size));

    let error = match state.k3_commit_solo(&qwen, &mut kv, &supervisor, plan) {
        Ok(_) => panic!("carry hidden must fail with one verifier row"),
        Err(error) => format!("{error:#}"),
    };

    assert!(
        error.starts_with("Qwen SlotAware MTP K3 carry hidden"),
        "{error}"
    );
    assert!(!kv.la_capture_active(), "settlement cleared capture");
    assert_slot_reset(&kv);
    assert_eq!(peer_digests(&kv), peers, "peer slots");
    assert!(state.pending_speculation_output.is_empty());
    assert!(!state.terminal_after_pending);
    assert_eq!(state.generated_tokens, generated);
    assert_eq!(verifier_hidden_bits(&state), hidden_before);
}

/// T-J1-5: the unit sequence a joint step runs per lane (cursors, restore,
/// settle, finish) leaves the committing slot byte-identical to the solo
/// commit over a twin cache, restores the slot's current DeltaNet state from
/// the carry row, installs the plan's ledger and carry hidden row, and
/// leaves every peer slot byte-identical.
#[test]
fn unit_commit_sequence_matches_solo_commit_and_mutates_only_its_slot() {
    let _gpu = crate::inference::hf2q_gpu_test_lock();
    let device = MlxDevice::new().expect("Metal device");
    let qwen = loaded_model(QwenSpeculationPolicy::Auto);
    let cfg = cache_cfg(&qwen);
    for (name, decisions) in [
        ("full", FULL),
        ("partial", PARTIAL),
        ("zero", ZERO),
        ("terminal miss at 2", TERMINAL_MISS_AT_2),
    ] {
        let mut solo_kv = verified_cache(&cfg, &device);
        let mut unit_kv = verified_cache(&cfg, &device);
        let mut solo = decode_state(&qwen, &device);
        let mut unit = decode_state(&qwen, &device);
        let supervisor = EngineSupervisor::new();
        let peers = peer_digests(&solo_kv);
        assert_eq!(peer_digests(&unit_kv), peers, "{name}: twin caches");
        let entry_state = digest(&solo_kv, SLOT);
        let solo_plan = k3_plan(
            decisions,
            hidden_rows(&device, K3_VERIFY_ROWS, qwen.hidden_size),
        );
        let unit_plan = k3_plan(
            decisions,
            hidden_rows(&device, K3_VERIFY_ROWS, qwen.hidden_size),
        );
        let expected = scripted_plan(decisions);
        let carry = expected.carry_hidden_row;
        let committed = PRIOR + expected.valid_input_tokens as u32;
        let carry_bits = hidden_row_bits(&solo_plan.hidden, carry, qwen.hidden_size);
        let partial = expected.valid_input_tokens < K3_VERIFY_ROWS;
        let restored: Vec<_> = (0..solo_kv.linear_attn.len())
            .map(|layer| capture_row(&solo_kv, layer, SLOT, carry))
            .collect();

        let solo_commit = solo
            .k3_commit_solo(&qwen, &mut solo_kv, &supervisor, solo_plan)
            .unwrap_or_else(|error| panic!("{name}: solo commit: {error:#}"));
        let owed = unit
            .k3_commit_cursors(&mut unit_kv, &unit_plan)
            .unwrap_or_else(|error| panic!("{name}: cursors: {error:#}"));
        if let Some(restore) = owed {
            unit_kv
                .rollback_la_to(restore.slot, restore.capture_row)
                .unwrap_or_else(|error| panic!("{name}: restore: {error:#}"));
        }
        unit.k3_commit_settle(&mut unit_kv, &unit_plan)
            .unwrap_or_else(|error| panic!("{name}: settle: {error:#}"));
        let unit_commit = unit
            .k3_commit_finish(&qwen, &mut unit_kv, unit_plan)
            .unwrap_or_else(|error| panic!("{name}: finish: {error:#}"));

        assert_eq!(solo_commit.committed, unit_commit, "{name}");
        assert_eq!(unit_commit.valid_input_tokens, expected.valid_input_tokens);
        if !partial {
            assert_eq!(
                solo_commit.rollback_elapsed,
                Duration::ZERO,
                "{name}: no restore"
            );
        }
        let solo_slot = digest(&solo_kv, SLOT);
        assert_eq!(solo_slot, digest(&unit_kv, SLOT), "{name}: slot digests");
        assert!(solo_slot
            .target_cursors
            .iter()
            .all(|&cursor| cursor == committed));
        assert_eq!(solo_slot.mtp_cursor, committed, "{name}");
        assert_eq!(solo_slot.parity, entry_state.parity, "{name}: parity");
        assert_eq!(
            solo_slot.attention, entry_state.attention,
            "{name}: K/V bytes"
        );
        for (layer, (state, captured)) in solo_slot.linear.iter().zip(&restored).enumerate() {
            let entry = &entry_state.linear[layer];
            let (conv, recurrent) = if partial {
                (&captured[0], &captured[1])
            } else {
                (&entry[0], &entry[2])
            };
            assert_eq!(&state[0], conv, "{name}: layer {layer} current conv");
            assert_eq!(
                &state[2], recurrent,
                "{name}: layer {layer} current recurrent"
            );
            assert_eq!(state[1], entry[1], "{name}: layer {layer} conv scratch");
            assert_eq!(
                state[3], entry[3],
                "{name}: layer {layer} recurrent scratch"
            );
        }
        assert!(
            !solo_kv.la_capture_active() && !unit_kv.la_capture_active(),
            "{name}"
        );
        assert_eq!(peer_digests(&solo_kv), peers, "{name}: solo peers");
        assert_eq!(peer_digests(&unit_kv), peers, "{name}: unit peers");
        for state in [&solo, &unit] {
            assert_eq!(state.pending_speculation_output, expected.output, "{name}");
            assert_eq!(
                state.terminal_after_pending, expected.terminal_after_pending,
                "{name}"
            );
            assert_eq!(
                verifier_hidden_bits(state),
                carry_bits,
                "{name}: carry hidden"
            );
        }
    }
}

/// T-P4b-5: full, partial and zero acceptance through the production solo
/// commit are byte-identical with the GPU restore switch on and off.
#[test]
fn gpu_restore_transaction_cases_match_host() {
    let _gpu = crate::inference::hf2q_gpu_test_lock();
    let device = MlxDevice::new().expect("Metal device");
    let qwen = loaded_model(QwenSpeculationPolicy::Auto);
    let cfg = cache_cfg(&qwen);
    let supervisor = EngineSupervisor::new();

    for (name, decisions) in [("full", FULL), ("partial", PARTIAL), ("zero", ZERO)] {
        let mut host_kv = verified_cache(&cfg, &device);
        let mut gpu_kv = verified_cache(&cfg, &device);
        gpu_kv.set_la_restore_route(LaRestoreRoute::Gpu);
        let mut host = decode_state(&qwen, &device);
        let mut gpu = decode_state(&qwen, &device);
        let host_plan = k3_plan(
            decisions,
            hidden_rows(&device, K3_VERIFY_ROWS, qwen.hidden_size),
        );
        let gpu_plan = k3_plan(
            decisions,
            hidden_rows(&device, K3_VERIFY_ROWS, qwen.hidden_size),
        );

        let host_commit = host
            .k3_commit_solo(&qwen, &mut host_kv, &supervisor, host_plan)
            .unwrap_or_else(|error| panic!("{name}: host commit: {error:#}"));
        let gpu_commit = gpu
            .k3_commit_solo(&qwen, &mut gpu_kv, &supervisor, gpu_plan)
            .unwrap_or_else(|error| panic!("{name}: GPU commit: {error:#}"));

        assert_eq!(gpu_commit.committed, host_commit.committed, "{name}");
        assert_eq!(digest(&gpu_kv, SLOT), digest(&host_kv, SLOT), "{name}");
        assert_eq!(peer_digests(&gpu_kv), peer_digests(&host_kv), "{name}");
        assert_eq!(
            gpu.pending_speculation_output,
            host.pending_speculation_output,
            "{name}"
        );
        assert_eq!(gpu.terminal_after_pending, host.terminal_after_pending, "{name}");
        assert_eq!(verifier_hidden_bits(&gpu), verifier_hidden_bits(&host), "{name}");
    }
}
