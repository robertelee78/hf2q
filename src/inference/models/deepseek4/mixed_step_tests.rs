//! ADR-063 S1: model-free mixed-step transaction isolation.
//!
//! One cooperative prefill cohort and two decoding nonmembers share one
//! scheduler step over real `Deepseek4Cache` allocations (no weights, no
//! kernels). Each transaction's physical writes are emulated on exactly the
//! rows its production plan (`CacheSpan` / `CacheStep`) names; the
//! production settler and publishers (`settle_prefill_cohort`,
//! `publish_verifier_cohort_after_gate`, `snapshot`/`restore`) decide what
//! becomes visible. Oracle: every nonmember's bytes, cursor and poison flag
//! equal an independent serial control that never saw the cohort; every
//! member either equals its serially committed control or keeps its start
//! cursor and is poisoned; a cancelled member rolls back to its anchor.
//! Kernel and shared-scratch isolation is proven on hardware by
//! `real_artifact_mixed_step_tests`.

use anyhow::Result;
use mlx_native::{MlxBuffer, MlxDevice};

use super::cache::{CacheSpan, Deepseek4Cache, Deepseek4CachePlan, LayerCache};
use super::cache_tests::config;
use super::decode_cohort::publish_verifier_cohort_after_gate;
use super::verifier_forward::settle_prefill_cohort;

const LANES: usize = 5;
/// Slot order interleaves members with decoders, so a selection that
/// confuses slot index with cohort index cannot pass by contiguity.
const MEMBERS: [usize; 3] = [0, 2, 3];
const NONMEMBERS: [usize; 2] = [1, 4];
/// Members share the protected agent anchor's compressor remainders
/// (mod-128 = 20, mod-4 = 0); the decoders complete ratio-4 and ratio-128
/// groups during the scenario.
const START: [usize; LANES] = [148, 125, 148, 148, 254];
const COHORT_ROWS: usize = 128;
const CONTEXT: usize = 512;
const STEPS: usize = 6;
const COHORT_STEP: usize = 2;
const CANCELLED: usize = 2;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Fault {
    /// Every submission completes and the commit gate accepts.
    Success,
    /// Encoding fails before any command buffer is submitted.
    PreSubmitFailure,
    /// Every submission completes, then the supervisor gate rejects.
    LateFailure,
    /// Execution fails after one layer's command buffers wrote member state.
    Poison,
    /// The cohort commits, then one member's client cancels and that member
    /// rolls back to its prompt-boundary anchor.
    Cancel,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Order {
    DecodeFirst,
    CohortFirst,
}

/// Small three-layer plan (uncompressed, ratio-4 with indexer, ratio-128)
/// shared with the serving-ledger half of the S1 fixtures.
pub(crate) fn host_cache_plan() -> Deepseek4CachePlan {
    Deepseek4CachePlan::for_context(&config(vec![0, 4, 128]), CONTEXT).unwrap()
}

fn row_bytes(buffer: &MlxBuffer) -> usize {
    buffer.shape()[1..].iter().product::<usize>() * buffer.dtype().size_of()
}

fn fill_rows(buffer: &mut MlxBuffer, row: usize, rows: usize, byte: u8) {
    let width = row_bytes(buffer);
    let bytes = buffer.as_mut_slice::<u8>().unwrap();
    bytes[row * width..(row + rows) * width].fill(byte);
}

fn fill_all(buffer: Option<&mut MlxBuffer>, byte: u8) {
    if let Some(buffer) = buffer {
        buffer.as_mut_slice::<u8>().unwrap().fill(byte);
    }
}

fn fill_states(layer: &mut LayerCache, byte: u8) {
    for state in [
        layer.main_kv_state.as_mut(),
        layer.main_score_state.as_mut(),
        layer.indexer_kv_state.as_mut(),
        layer.indexer_score_state.as_mut(),
    ] {
        fill_all(state, byte);
    }
}

fn decode_salt(lane: usize, step: usize) -> u8 {
    (lane * 37 + step * 11 + 1) as u8
}

fn cohort_salt(lane: usize) -> u8 {
    (lane * 29 + 0x80) as u8
}

/// Lane-specific canary in every physical byte, then a warm cursor.
fn fixture(device: &MlxDevice) -> Vec<Deepseek4Cache> {
    let plan = host_cache_plan();
    (0..LANES)
        .map(|lane| {
            let mut cache = Deepseek4Cache::allocate(&plan, device.clone()).unwrap();
            let canary = 0x40 + lane as u8;
            for layer in cache.layers_mut() {
                fill_all(Some(&mut layer.attention_kv), canary);
                fill_all(layer.indexer_kv.as_mut(), canary);
                fill_states(layer, canary ^ 0x0f);
            }
            cache.commit_prefill(0, START[lane]).unwrap();
            cache
        })
        .collect()
}

/// Every physical byte a lane owns, visible or not.
fn all_bytes(cache: &Deepseek4Cache) -> Vec<Vec<u8>> {
    cache
        .layers()
        .iter()
        .flat_map(|layer| {
            std::iter::once(&layer.attention_kv)
                .chain(layer.indexer_kv.iter())
                .chain(layer.main_kv_state.iter())
                .chain(layer.main_score_state.iter())
                .chain(layer.indexer_kv_state.iter())
                .chain(layer.indexer_score_state.iter())
                .map(|buffer| buffer.as_slice::<u8>().unwrap().to_vec())
        })
        .collect()
}

fn prefix_rows(buffer: &MlxBuffer, rows: usize) -> Vec<u8> {
    let bytes = buffer.as_slice::<u8>().unwrap();
    bytes[..rows * row_bytes(buffer)].to_vec()
}

/// Bytes an attention reader can observe at the cache cursor.
fn visible_bytes(cache: &Deepseek4Cache) -> Vec<Vec<u8>> {
    let position = cache.position();
    let mut bytes = Vec::new();
    for (layer, plan) in cache.layers().iter().zip(&cache.plan.layers) {
        let window_rows = position.min(layer.window_kv.shape()[0]);
        bytes.push(prefix_rows(&layer.window_kv, window_rows));
        let compressed_rows = match plan.compress_ratio {
            0 => 0,
            ratio => position / ratio as usize,
        };
        for buffer in layer.compressed_kv.iter().chain(layer.indexer_kv.iter()) {
            bytes.push(prefix_rows(buffer, compressed_rows));
        }
        for state in layer
            .main_kv_state
            .iter()
            .chain(layer.main_score_state.iter())
            .chain(layer.indexer_kv_state.iter())
            .chain(layer.indexer_score_state.iter())
        {
            bytes.push(state.as_slice::<u8>().unwrap().to_vec());
        }
    }
    bytes
}

fn select<'a>(caches: &'a mut [Deepseek4Cache], lanes: &[usize]) -> Vec<&'a mut Deepseek4Cache> {
    caches
        .iter_mut()
        .enumerate()
        .filter(|(lane, _)| lanes.contains(lane))
        .map(|(_, cache)| cache)
        .collect()
}

/// Write exactly the rows one decode token's plan names, as the body would.
fn emulate_decode_writes(cache: &mut Deepseek4Cache, byte: u8) -> usize {
    let step = cache.plan_next_step().unwrap();
    for (layer, plan) in cache.layers_mut().iter_mut().zip(&step.layers) {
        fill_rows(&mut layer.window_kv, plan.window_write_slot, 1, byte);
        if let (Some(slot), Some(buffer)) =
            (plan.compressed_write_slot, layer.compressed_kv.as_mut())
        {
            fill_rows(buffer, slot, 1, byte);
        }
        if let (Some(slot), Some(buffer)) = (plan.indexer_write_slot, layer.indexer_kv.as_mut()) {
            fill_rows(buffer, slot, 1, byte);
        }
        fill_states(layer, byte);
    }
    step.position
}

/// Write the rows a cooperative transaction's first `layers` layers name.
fn emulate_prefill_writes(cache: &mut Deepseek4Cache, span: &CacheSpan, layers: usize, byte: u8) {
    let executed = &span.layers[..layers.min(span.layers.len())];
    for (layer, plan) in cache.layers_mut().iter_mut().zip(executed) {
        let capacity = layer.window_kv.shape()[0];
        for offset in 0..plan.window_write_count {
            let row = (plan.window_write_start + offset) % capacity;
            fill_rows(&mut layer.window_kv, row, 1, byte);
        }
        if let Some(buffer) = layer.compressed_kv.as_mut() {
            fill_rows(
                buffer,
                plan.compressed_write_start,
                plan.compressed_count,
                byte,
            );
        }
        if let Some(buffer) = layer.indexer_kv.as_mut() {
            fill_rows(buffer, plan.indexer_write_start, plan.indexer_count, byte);
        }
        fill_states(layer, byte);
    }
}

/// Production joint-decode publication for the decoding nonmembers.
fn mixed_decode(caches: &mut [Deepseek4Cache], step: usize) {
    let positions =
        NONMEMBERS.map(|lane| emulate_decode_writes(&mut caches[lane], decode_salt(lane, step)));
    let mut decoders = select(caches, &NONMEMBERS);
    publish_verifier_cohort_after_gate(&mut decoders, &positions, || Ok(())).unwrap();
}

/// Independent serial execution: each decoder commits its own token.
fn control_decode(caches: &mut [Deepseek4Cache], step: usize) {
    for lane in NONMEMBERS {
        let position = emulate_decode_writes(&mut caches[lane], decode_salt(lane, step));
        caches[lane].commit_step(position).unwrap();
    }
}

/// One cooperative transaction settled by the production settler.
fn mixed_cohort(caches: &mut [Deepseek4Cache], fault: Fault) -> Result<()> {
    let spans = MEMBERS.map(|lane| caches[lane].plan_prefill(COHORT_ROWS).unwrap());
    let executed_layers = match fault {
        Fault::PreSubmitFailure => 0,
        Fault::Poison => 1,
        Fault::Success | Fault::LateFailure | Fault::Cancel => usize::MAX,
    };
    for (&lane, span) in MEMBERS.iter().zip(&spans) {
        emulate_prefill_writes(&mut caches[lane], span, executed_layers, cohort_salt(lane));
    }
    let execution = match fault {
        Fault::PreSubmitFailure => (Err(anyhow::anyhow!("synthetic encode failure")), false),
        Fault::Poison => (Err(anyhow::anyhow!("synthetic layer-0 completion failure")), true),
        Fault::Success | Fault::LateFailure | Fault::Cancel => (Ok(()), true),
    };
    let mut members = select(caches, &MEMBERS);
    settle_prefill_cohort(&mut members, &spans, execution, || {
        anyhow::ensure!(
            fault != Fault::LateFailure,
            "synthetic late supervisor rejection"
        );
        Ok(())
    })
}

/// Independent serial execution of the members that should commit.
fn control_cohort(caches: &mut [Deepseek4Cache], fault: Fault) {
    if !matches!(fault, Fault::Success | Fault::Cancel) {
        return;
    }
    for lane in MEMBERS {
        if fault == Fault::Cancel && lane == CANCELLED {
            continue;
        }
        let span = caches[lane].plan_prefill(COHORT_ROWS).unwrap();
        emulate_prefill_writes(&mut caches[lane], &span, usize::MAX, cohort_salt(lane));
        let (start, rows) = (span.start_position, span.token_count);
        caches[lane].commit_prefill(start, rows).unwrap();
    }
}

fn assert_member_outcome(
    label: &str,
    mixed: &[Deepseek4Cache],
    control: &[Deepseek4Cache],
    fault: Fault,
) {
    for lane in MEMBERS {
        let cache = &mixed[lane];
        let committed = fault == Fault::Success || (fault == Fault::Cancel && lane != CANCELLED);
        let clean = committed || fault == Fault::PreSubmitFailure || fault == Fault::Cancel;
        if clean {
            assert!(!cache.is_poisoned(), "{label} member {lane} poisoned");
            assert_eq!(
                cache.position(),
                control[lane].position(),
                "{label} member {lane} cursor"
            );
            assert!(
                visible_bytes(cache) == visible_bytes(&control[lane]),
                "{label} member {lane} visible state differs from serial execution"
            );
        } else {
            assert!(cache.is_poisoned(), "{label} member {lane} not poisoned");
            assert_eq!(
                cache.position(),
                START[lane],
                "{label} member {lane} published"
            );
            assert!(cache.plan_prefill(1).is_err());
            assert!(cache.plan_next_step().is_err());
        }
        let expected = if committed {
            START[lane] + COHORT_ROWS
        } else {
            START[lane]
        };
        assert_eq!(cache.position(), expected, "{label} member {lane} cursor");
    }
    if fault == Fault::PreSubmitFailure {
        for lane in MEMBERS {
            assert!(
                all_bytes(&mixed[lane]) == all_bytes(&control[lane]),
                "{label} pre-submission failure wrote member {lane}"
            );
        }
    }
}

fn run(fault: Fault, order: Order) {
    let label = format!("{fault:?}/{order:?}");
    let device = MlxDevice::new().unwrap();
    let mut mixed = fixture(&device);
    let mut control = fixture(&device);
    for step in 0..STEPS {
        if step == COHORT_STEP {
            let anchor = mixed[CANCELLED].snapshot().unwrap();
            if order == Order::DecodeFirst {
                mixed_decode(&mut mixed, step);
            }
            let outcome = mixed_cohort(&mut mixed, fault);
            if order == Order::CohortFirst {
                mixed_decode(&mut mixed, step);
            }
            control_decode(&mut control, step);
            control_cohort(&mut control, fault);
            let expected_context = match fault {
                Fault::Success | Fault::Cancel => None,
                Fault::PreSubmitFailure => Some("failed before GPU submission"),
                Fault::LateFailure => Some("rejected before cache publication"),
                Fault::Poison => Some("partially executed; caches poisoned"),
            };
            match (outcome, expected_context) {
                (Ok(()), None) => {}
                (Err(error), Some(context)) => assert!(
                    format!("{error:#}").contains(context),
                    "{label} unexpected cohort error: {error:#}"
                ),
                (outcome, _) => panic!("{label} unexpected cohort outcome: {outcome:?}"),
            }
            if fault == Fault::Cancel {
                mixed[CANCELLED].restore(&anchor).unwrap();
            }
            assert_member_outcome(&label, &mixed, &control, fault);
        } else {
            mixed_decode(&mut mixed, step);
            control_decode(&mut control, step);
        }
        for lane in NONMEMBERS {
            assert!(
                !mixed[lane].is_poisoned(),
                "{label} step {step} decoder {lane} poisoned"
            );
            assert_eq!(
                mixed[lane].position(),
                control[lane].position(),
                "{label} step {step} decoder {lane} cursor"
            );
            assert!(
                all_bytes(&mixed[lane]) == all_bytes(&control[lane]),
                "{label} step {step} decoder {lane} bytes differ from independent execution"
            );
        }
    }
    for lane in NONMEMBERS {
        assert_eq!(mixed[lane].position(), START[lane] + STEPS);
    }
    for lane in MEMBERS {
        if mixed[lane].is_poisoned() {
            mixed[lane].reset().unwrap();
            assert_eq!(mixed[lane].position(), 0);
            assert!(mixed[lane].plan_prefill(COHORT_ROWS).is_ok());
        }
    }
}

#[test]
fn mixed_step_cohort_outcomes_never_reach_decoding_nonmembers() {
    let _gpu = crate::inference::hf2q_gpu_test_lock();
    for fault in [
        Fault::Success,
        Fault::PreSubmitFailure,
        Fault::LateFailure,
        Fault::Poison,
        Fault::Cancel,
    ] {
        for order in [Order::DecodeFirst, Order::CohortFirst] {
            run(fault, order);
        }
    }
}

/// Isolation is a property of disjoint membership. A lane drafted into both
/// transactions of one step fails decode prevalidation, and the atomic
/// decode publisher then poisons every lane of that transaction, including
/// an innocent decoder. Serving must keep the sets disjoint upstream.
#[test]
fn overlapping_membership_poisons_an_innocent_decoder() {
    let _gpu = crate::inference::hf2q_gpu_test_lock();
    let device = MlxDevice::new().unwrap();
    let mut caches = fixture(&device);
    let (member, decoder, bystander) = (MEMBERS[0], NONMEMBERS[0], NONMEMBERS[1]);
    let overlapping = [member, decoder];
    let untouched = all_bytes(&caches[bystander]);
    // A policy that freezes its decode set at step start plans before the
    // cohort publishes.
    let positions =
        overlapping.map(|lane| emulate_decode_writes(&mut caches[lane], decode_salt(lane, 0)));
    mixed_cohort(&mut caches, Fault::Success).unwrap();
    assert_eq!(caches[member].position(), START[member] + COHORT_ROWS);

    let mut decoders = select(&mut caches, &overlapping);
    let error = publish_verifier_cohort_after_gate(&mut decoders, &positions, || Ok(()))
        .expect_err("stale overlapping member must fail decode prevalidation");
    assert!(format!("{error:#}").contains("prevalidate"));
    assert!(caches[decoder].is_poisoned());
    assert_eq!(caches[decoder].position(), START[decoder]);
    assert!(caches[member].is_poisoned());
    assert!(!caches[bystander].is_poisoned());
    assert!(all_bytes(&caches[bystander]) == untouched);
}
