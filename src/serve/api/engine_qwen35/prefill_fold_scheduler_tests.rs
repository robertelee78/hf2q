//! The cue fold's scheduler wiring on the synthetic model (default-run; no
//! model file): `Qwen35PrefillState::advance` with
//! `HF2Q_QWEN_PREFILL_FOLD_CUE` on runs the content chunk and the cue as
//! one transaction that returns `Ready` with the stable-boundary
//! checkpoint, and leaves what the two transactions leave; the conditions
//! under which `cue_fold_prefix_rows` keeps the two forwards; and a second
//! request that restores the folded checkpoint and diverges behind it.
//!
//! The switches come from a thread-local test override, not the process
//! environment. The DFlash2 lane branch of the folded transaction needs a
//! draft runtime and is not reached here; its two-range ingest is covered
//! by the ingest gates in `inference::spec_decode::dflash2::staged`.

use std::collections::BTreeMap;
use std::sync::atomic::Ordering;

use mlx_native::MlxDevice;

use super::{
    anchor_store, HybridPromptCache, Qwen35LoadedModel, Qwen35PrefillAdvance, Qwen35PrefillState,
};
use crate::inference::glp::bind::BoundGlp;
use crate::inference::glp::reader::{GlpHookPoint, GlpMode, GlpVector};
use crate::inference::models::qwen35::kv_cache::{HybridKvCache, HybridKvSlotAnchor};
use crate::inference::models::qwen35::prefill_prepared::{PrefillSwitches, SwitchesForTest};
use crate::inference::models::qwen35::prefill_prepared_tests::observable_synthetic_model;
use crate::serve::api::engine::SamplingParams;
use crate::serve::api::engine_supervisor::EngineSupervisor;
use crate::serve::multi_seq_kv::SlotId;

const SLOTS: u32 = 2;
const CAPACITY: u32 = 256;
/// Larger than any prompt here: a chunk ends at the stable boundary or at
/// the prompt's end.
const CHUNK: usize = 720;
const SLOT: SlotId = SlotId(1);

fn token(stream: u32, position: usize) -> u32 {
    (stream * 131 + position as u32 * 37 + 11) % 512
}

fn tokens(stream: u32, range: std::ops::Range<usize>) -> Vec<u32> {
    range.map(|position| token(stream, position)).collect()
}

fn bits(values: &[f32]) -> Vec<u32> {
    values.iter().map(|value| value.to_bits()).collect()
}

fn switches(fold_cue: bool) -> SwitchesForTest {
    SwitchesForTest::set(PrefillSwitches {
        prepared: true,
        fold_cue,
        joint: false,
        wait_layers: 1,
        keep_fresh: false,
    })
}

/// A loaded model around the observable synthetic model, boxed so that the
/// address the GPU cache is primed for does not move.
fn loaded(device: &MlxDevice) -> Box<Qwen35LoadedModel> {
    let model = observable_synthetic_model(device);
    let (hidden_size, vocab_size) = (
        model.cfg.hidden_size as usize,
        model.cfg.vocab_size as usize,
    );
    let qwen = Box::new(Qwen35LoadedModel {
        model,
        tokenizer: tokenizers::Tokenizer::new(tokenizers::models::bpe::BPE::default()),
        chat_template: "{{ messages }}".to_string(),
        model_id: "prefill-fold-scheduler-test".to_string(),
        model_path: std::path::PathBuf::from("/tmp/prefill-fold-scheduler-test.gguf"),
        eos_token_ids: vec![151_645],
        hidden_size,
        vocab_size,
        context_length: Some(CAPACITY as usize),
        quant_type: Some("Q4_K".to_string()),
        load_duration: std::time::Duration::from_millis(1),
        provenance: crate::core::provenance::Provenance::External,
        expected_projector_sha256: None,
        vision_projector_profile: None,
        vision_deepstack_output_count: None,
        vision_special_tokens_present: false,
        prompt_cache: HybridPromptCache::new(),
        lcp_registry: crate::serve::kv_persist::lcp_registry::LcpRegistry::new(1),
        kv_metrics_sink: None,
        disk_persistor: None,
        lcp_hydrated_for_cfg: std::collections::HashSet::new(),
        tq_kv_active: true,
        persistent_kv_cache: None,
        dflash2: None,
        speculation: crate::serve::api::qwen35_speculation::QwenSpeculationController::new(
            crate::serve::api::qwen35_speculation::QwenSpeculationPolicy::Off,
        ),
        slot_spill_budget_bytes: 0,
    });
    qwen.model
        .ensure_gpu_cache_primed()
        .expect("prime the GPU cache at the model's final address");
    qwen
}

fn new_cache(qwen: &Qwen35LoadedModel) -> HybridKvCache {
    qwen.model
        .with_gpu_cache_mut(|device, _registry| {
            HybridKvCache::new_with_options(&qwen.model.cfg, device, CAPACITY, SLOTS, true)
        })
        .expect("allocate TQ cache")
}

fn begin(
    qwen: &Qwen35LoadedModel,
    kv: &mut HybridKvCache,
    prompt: &[u32],
    stable_prefix: Option<usize>,
    cached_tokens: usize,
) -> Qwen35PrefillState {
    let params = SamplingParams {
        max_tokens: 1,
        stable_prompt_prefix_tokens: stable_prefix,
        ..SamplingParams::default()
    };
    Qwen35PrefillState::begin(
        prompt.to_vec(),
        params,
        None,
        kv,
        SLOT,
        cached_tokens,
        None,
        None,
        None,
        qwen.hidden_size,
        None,
    )
    .unwrap_or_else(|error| panic!("prefill admission: {error:#}"))
}

/// What one request's bounded prefill hands the scheduler.
struct Served {
    /// `advance` calls until `Ready`.
    transactions: usize,
    /// `(transaction index, prompt rows, checkpoint)` of every checkpoint.
    checkpoints: Vec<(usize, Vec<u32>, Vec<u32>, HybridKvSlotAnchor)>,
    /// Whether the `Ready` transaction carried a checkpoint.
    ready_with_checkpoint: bool,
    prefill_logits: Vec<u32>,
}

fn serve(
    qwen: &mut Qwen35LoadedModel,
    kv: &mut HybridKvCache,
    prompt: &[u32],
    stable_prefix: Option<usize>,
    cached_tokens: usize,
    supervisor: &EngineSupervisor,
) -> Served {
    let mut pending = Some(begin(qwen, kv, prompt, stable_prefix, cached_tokens));
    let mut checkpoints = Vec::new();
    let mut transactions = 0;
    loop {
        let advance = pending
            .take()
            .expect("a pending prefill")
            .advance(qwen, None, kv, CHUNK, supervisor, None)
            .unwrap_or_else(|error| panic!("bounded prefill: {error:#}"));
        let (checkpoint, ready) = match advance {
            Qwen35PrefillAdvance::Pending {
                state, checkpoint, ..
            } => {
                pending = Some(state);
                (checkpoint, None)
            }
            Qwen35PrefillAdvance::Ready {
                prefill_logits,
                checkpoint,
                ..
            } => (checkpoint, Some(prefill_logits)),
        };
        let carried = checkpoint.is_some();
        if let Some(checkpoint) = checkpoint {
            checkpoints.push((
                transactions,
                checkpoint.prompt_tokens,
                bits(&checkpoint.prefill_logits),
                checkpoint.kv,
            ));
        }
        transactions += 1;
        if let Some(prefill_logits) = ready {
            return Served {
                transactions,
                checkpoints,
                ready_with_checkpoint: carried,
                prefill_logits: bits(&prefill_logits),
            };
        }
    }
}

/// A first request whose prompt has a stable boundary, then a second request
/// that restores the boundary checkpoint and diverges behind it.
struct Conversation {
    first: Served,
    /// The checkpoint's cursors and DeltaNet words.
    checkpoint_words: (Vec<u32>, Vec<u32>),
    second_logits: Vec<u32>,
    /// Cursors and DeltaNet words after the second request's prefill.
    second_state: (Vec<u32>, Vec<u32>),
}

/// One conversation on a fresh cache of `qwen`. Both arms of a comparison
/// run on the one loaded model: the thread's buffer pool belongs to its
/// device.
fn converse(
    qwen: &mut Qwen35LoadedModel,
    supervisor: &EngineSupervisor,
    fold_cue: bool,
    prefix: usize,
    cue: usize,
) -> Conversation {
    const SUFFIX: usize = 11;
    let _switches = switches(fold_cue);
    let mut kv = new_cache(qwen);
    let prompt = tokens(1, 0..prefix + cue);
    let first = serve(qwen, &mut kv, &prompt, Some(prefix), 0, supervisor);
    let (_, _, _, anchor) = first.checkpoints.last().expect("a boundary checkpoint");
    let words = anchor.test_words();
    let plan = kv
        .plan_slot_anchor_restore(SLOT, anchor)
        .expect("restore plan");
    kv.apply_slot_anchor_restore(plan).expect("restore");
    let mut diverging = prompt[..prefix].to_vec();
    diverging.extend(tokens(9, prefix..prefix + SUFFIX));
    let second = serve(qwen, &mut kv, &diverging, None, prefix, supervisor);
    let after = kv
        .snapshot_slot_anchor(SLOT, prefix + SUFFIX)
        .expect("slot state after the second request")
        .test_words();
    Conversation {
        first,
        checkpoint_words: (words.3, words.5),
        second_logits: second.prefill_logits,
        second_state: (after.3, after.5),
    }
}

#[test]
fn a_folded_transaction_is_ready_with_the_checkpoint_the_two_transactions_take() {
    let _gpu = crate::inference::hf2q_gpu_test_lock();
    let Ok(device) = MlxDevice::new() else {
        return;
    };
    let mut qwen = loaded(&device);
    let supervisor = EngineSupervisor::new();
    for (prefix, cue) in [(40, 5), (97, 8)] {
        let snapshots = || {
            anchor_store::ANCHOR_TELEMETRY
                .snapshot_count
                .load(Ordering::Relaxed)
        };
        let before = snapshots();
        let two = converse(&mut qwen, &supervisor, false, prefix, cue);
        let between = snapshots();
        let folded = converse(&mut qwen, &supervisor, true, prefix, cue);
        let after = snapshots();

        // Two transactions: the content chunk hands over the checkpoint,
        // the cue chunk is ready without one.
        assert_eq!(two.first.transactions, 2, "{prefix}+{cue}");
        assert!(!two.first.ready_with_checkpoint);
        assert_eq!(two.first.checkpoints.len(), 1);
        assert_eq!(two.first.checkpoints[0].0, 0);
        // One folded transaction: ready, with the checkpoint.
        assert_eq!(folded.first.transactions, 1, "{prefix}+{cue}");
        assert!(folded.first.ready_with_checkpoint);
        assert_eq!(folded.first.checkpoints.len(), 1);

        let (two_checkpoint, folded_checkpoint) =
            (&two.first.checkpoints[0], &folded.first.checkpoints[0]);
        assert_eq!(two_checkpoint.1, tokens(1, 0..prefix));
        assert_eq!(folded_checkpoint.1, two_checkpoint.1, "checkpoint prompt");
        assert!(
            folded_checkpoint.2 == two_checkpoint.2,
            "{prefix}+{cue}: boundary logits differ"
        );
        assert!(
            folded.checkpoint_words == two.checkpoint_words,
            "{prefix}+{cue}: the boundary checkpoint differs"
        );
        assert!(
            folded.first.prefill_logits == two.first.prefill_logits,
            "{prefix}+{cue}: the prompt's last-row logits differ"
        );
        // The request that restored the checkpoint and diverged.
        assert!(
            folded.second_logits == two.second_logits,
            "{prefix}+{cue}: logits behind the restored folded checkpoint differ"
        );
        assert!(
            folded.second_state == two.second_state,
            "{prefix}+{cue}: slot state behind the restored folded checkpoint differs"
        );
        // Each arm took one checkpoint and counted it (other tests of this
        // process may count too, so at least one each).
        assert!(between > before && after > between, "snapshot metric");
    }
}

#[test]
fn the_scheduler_keeps_two_forwards_where_the_fold_does_not_apply() {
    let _gpu = crate::inference::hf2q_gpu_test_lock();
    let Ok(device) = MlxDevice::new() else {
        return;
    };
    let mut qwen = loaded(&device);
    let mut kv = new_cache(&qwen);
    /// The content rows the scheduler would fold, if it folds.
    fn folds(
        qwen: &Qwen35LoadedModel,
        kv: &mut HybridKvCache,
        prefix: usize,
        cue: usize,
        mtp_prefill: bool,
        headroom: Option<usize>,
    ) -> Option<usize> {
        let prompt = tokens(1, 0..prefix + cue);
        let state = begin(qwen, kv, &prompt, Some(prefix), 0);
        state
            .cue_fold_prefix_rows(qwen, kv, prefix, false, mtp_prefill, headroom)
            .expect("fold decision")
            .map(|(rows, _image)| rows)
    }
    {
        let _switches = switches(true);
        assert_eq!(folds(&qwen, &mut kv, 40, 5, false, None), Some(40));
        assert_eq!(folds(&qwen, &mut kv, 40, 32, false, None), Some(40));
        // A short content chunk the cue would carry past 32 rows.
        assert_eq!(folds(&qwen, &mut kv, 31, 5, false, None), None);
        assert_eq!(folds(&qwen, &mut kv, 20, 5, false, None), Some(20));
        // One-row segments and a cue longer than 32 rows.
        assert_eq!(folds(&qwen, &mut kv, 40, 1, false, None), None);
        assert_eq!(folds(&qwen, &mut kv, 1, 5, false, None), None);
        assert_eq!(folds(&qwen, &mut kv, 40, 33, false, None), None);
        // The MTP catch-up prefill, and a checkpoint over its quota.
        assert_eq!(folds(&qwen, &mut kv, 40, 5, true, None), None);
        assert_eq!(folds(&qwen, &mut kv, 40, 5, false, Some(0)), None);
        assert_eq!(
            folds(&qwen, &mut kv, 40, 5, false, Some(usize::MAX)),
            Some(40)
        );
        // A bound GLP vector (`hf2q serve --glp`): two forwards, no error.
        let width = qwen.hidden_size;
        qwen.model.glp = Some(BoundGlp {
            vector: GlpVector {
                mode: GlpMode::Add,
                hook_point: GlpHookPoint::ResidualStreamPostLayer,
                derived_at: None,
                alpha_default: 1.0,
                rank: 1,
                layers: BTreeMap::from([(3_u32, vec![0.25_f32; width])]),
                width,
                content_sha256: None,
                method: None,
                base_model_name: None,
            },
            alpha: 1.0,
            device_directions: BTreeMap::new(),
        });
        assert_eq!(folds(&qwen, &mut kv, 40, 5, false, None), None);
        qwen.model.glp = None;
        assert_eq!(folds(&qwen, &mut kv, 40, 5, false, None), Some(40));
    }
    {
        // The switch off, and the fold without the prepared kernels.
        let _switches = switches(false);
        assert_eq!(folds(&qwen, &mut kv, 40, 5, false, None), None);
    }
    let _switches = SwitchesForTest::set(PrefillSwitches {
        prepared: false,
        fold_cue: true,
        joint: false,
        wait_layers: 1,
        keep_fresh: false,
    });
    assert_eq!(folds(&qwen, &mut kv, 40, 5, false, None), None);
}
