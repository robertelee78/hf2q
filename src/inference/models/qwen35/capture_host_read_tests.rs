//! Metal gates for the host reads of the residual stream inside the layer
//! loop of `forward_gpu_impl`: the activation capture (DFlash target layers,
//! calibration) and the calibration residual capture.
//!
//! A multi-row forward of at most 32 rows batches its layer waits
//! (`HF2Q_FFN_TERMINAL_K_BATCH`, default 8), and the layer output lives in a
//! two-slot ring. A host copy taken at a layer that is not a wait boundary
//! therefore returned whatever the ring slot held: an earlier layer's
//! residual stream. These gates pin the captured rows to the rows of the
//! same forward with a wait after every layer.
//!
//! The model is the synthetic native K-quant model of the joint verify
//! tests (eight layers, so with the default batch only the last layer is a
//! boundary).

use std::sync::{Mutex, MutexGuard};

use mlx_native::MlxDevice;

use super::joint_verify_tests::{padded_policy, synthetic_model};
use super::kv_cache::HybridKvCache;
use super::model::Qwen35Model;
use super::spec_decode::positions_for_range;
use crate::core::traits::activation_capture::LayerActivations;
use crate::inference::spec_decode::dflash::hidden_capture::DFlashCaptureSession;
use crate::serve::multi_seq_kv::SlotId;

const K_BATCH_ENV: &str = "HF2Q_FFN_TERMINAL_K_BATCH";
/// Layers that are not a wait boundary at the default batch of eight, plus
/// the last layer, which always is.
const TARGET_LAYERS: [usize; 4] = [1, 4, 6, 7];
const CAPACITY: u32 = 128;

/// Serializes this module's edits of the K-batch variable. The forwards
/// that read it run under the GPU test lock the callers already hold.
static K_BATCH: Mutex<()> = Mutex::new(());

/// Runs with a wait after every layer while alive.
struct WaitEveryLayer {
    _guard: MutexGuard<'static, ()>,
}

impl WaitEveryLayer {
    fn enter() -> Self {
        let guard = K_BATCH
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        std::env::set_var(K_BATCH_ENV, "1");
        Self { _guard: guard }
    }
}

impl Drop for WaitEveryLayer {
    fn drop(&mut self) {
        std::env::remove_var(K_BATCH_ENV);
    }
}

fn tokens(start: usize, len: usize) -> Vec<u32> {
    (start..start + len)
        .map(|position| (position as u32 * 37 + 11) % 512)
        .collect()
}

fn cache(model: &Qwen35Model) -> HybridKvCache {
    model
        .with_gpu_cache_mut(|device, _registry| {
            HybridKvCache::new_with_options(&model.cfg, device, CAPACITY, 1, true)
        })
        .expect("allocate TQ cache")
}

fn bits(values: &[f32]) -> Vec<u32> {
    values.iter().map(|value| value.to_bits()).collect()
}

/// Prefill `history` rows, then capture `TARGET_LAYERS` over the next `rows`
/// rows. Returns the captured rows and the last-row logits, as bits.
fn dflash_capture(model: &Qwen35Model, history: usize, rows: usize) -> (Vec<u32>, Vec<u32>) {
    let mut cache = cache(model);
    if history > 0 {
        model
            .forward_gpu_last_logits(
                &tokens(0, history),
                &positions_for_range(0, history),
                &mut cache,
                SlotId(0),
            )
            .expect("history prefill");
    }
    let hidden = model.cfg.hidden_size as usize;
    let mut session = DFlashCaptureSession::new(TARGET_LAYERS.to_vec(), rows, hidden, false);
    let (logits, _hidden) = model
        .forward_gpu_last_logits_with_hidden_dflash(
            &tokens(history, rows),
            &positions_for_range(history as i32, rows),
            &mut cache,
            Some(&mut session),
            SlotId(0),
        )
        .expect("captured chunk");
    (bits(&session.hidden_output), bits(&logits))
}

fn with_synthetic_model(run: impl FnOnce(&Qwen35Model)) {
    let _gpu = crate::inference::hf2q_gpu_test_lock();
    let Ok(device) = MlxDevice::new() else {
        return;
    };
    assert!(
        std::env::var_os(K_BATCH_ENV).is_none(),
        "{K_BATCH_ENV} must be unset: these gates compare the default batch against 1"
    );
    let model = synthetic_model(&device, padded_policy(), false);
    // `synthetic_model` primed the thread's GPU cache for the model's
    // address before the move out of it.
    model
        .ensure_gpu_cache_primed()
        .expect("prime the GPU cache at the model's final address");
    run(&model);
}

#[test]
fn short_chunk_target_layer_capture_equals_the_wait_every_layer_capture() {
    with_synthetic_model(|model| {
        let hidden = model.cfg.hidden_size as usize;
        // A fresh chunk, the generation cue after a prompt, and resumed
        // chunks on both sides of the direct TQ attention route's 32 rows.
        for (history, rows) in [(0usize, 20usize), (20, 5), (20, 31), (40, 2), (17, 32)] {
            let batched = dflash_capture(model, history, rows);
            let waited = {
                let _wait = WaitEveryLayer::enter();
                dflash_capture(model, history, rows)
            };
            assert!(
                batched.1 == waited.1,
                "history {history} rows {rows}: last-row logits differ"
            );
            for (index, layer) in TARGET_LAYERS.iter().enumerate() {
                let span = index * rows * hidden..(index + 1) * rows * hidden;
                let differing = batched.0[span.clone()]
                    .iter()
                    .zip(&waited.0[span])
                    .filter(|(a, b)| a != b)
                    .count();
                assert_eq!(
                    differing,
                    0,
                    "history {history} rows {rows}: layer {layer} capture differs from the \
                     wait-every-layer capture in {differing} of {} elements",
                    rows * hidden
                );
            }
        }
    });
}

#[test]
fn short_prompt_full_activation_capture_equals_the_wait_every_layer_capture() {
    with_synthetic_model(|model| {
        let layers = model.cfg.num_hidden_layers as usize;
        let hidden = model.cfg.hidden_size as usize;
        let rows = 20usize;
        let capture = |model: &Qwen35Model| {
            let mut acts = LayerActivations {
                layer_inputs: Vec::with_capacity(layers),
                layer_outputs: Vec::with_capacity(layers),
                num_layers: layers as u32,
                seq_len: rows as u32,
                hidden_size: hidden as u32,
                target_layer_filter: None,
            };
            model
                .forward_gpu_with_capture(
                    &tokens(0, rows),
                    &positions_for_range(0, rows),
                    &mut cache(model),
                    &mut acts,
                )
                .expect("captured prompt");
            acts.validate().expect("capture shape");
            acts
        };
        let batched = capture(model);
        let waited = {
            let _wait = WaitEveryLayer::enter();
            capture(model)
        };
        for layer in 0..layers {
            assert!(
                bits(&batched.layer_inputs[layer]) == bits(&waited.layer_inputs[layer]),
                "layer {layer}: captured input differs from the wait-every-layer capture"
            );
            assert!(
                bits(&batched.layer_outputs[layer]) == bits(&waited.layer_outputs[layer]),
                "layer {layer}: captured output differs from the wait-every-layer capture"
            );
            if layer + 1 < layers {
                assert!(
                    bits(&batched.layer_outputs[layer]) == bits(&batched.layer_inputs[layer + 1]),
                    "layer {layer}: its output is not the next layer's input"
                );
            }
        }
    });
}

#[test]
fn short_prompt_residual_capture_equals_the_wait_every_layer_capture() {
    with_synthetic_model(|model| {
        let rows = 20usize;
        let capture = |model: &Qwen35Model, layer: usize| {
            bits(
                &model
                    .forward_gpu_capture_residual_layer(
                        &tokens(0, rows),
                        &positions_for_range(0, rows),
                        &mut cache(model),
                        layer,
                        SlotId(0),
                    )
                    .expect("residual capture"),
            )
        };
        for layer in TARGET_LAYERS {
            let batched = capture(model, layer);
            let waited = {
                let _wait = WaitEveryLayer::enter();
                capture(model, layer)
            };
            assert!(
                batched == waited,
                "layer {layer}: residual capture differs from the wait-every-layer capture"
            );
        }
    });
}

/// A one-row forward commits every layer without a wait, whatever the
/// batch. Its last captured layer must be the residual stream the forward
/// returns, which is read after the forward's terminal wait.
#[test]
fn one_row_capture_of_the_last_layer_is_the_returned_residual() {
    with_synthetic_model(|model| {
        let layers = model.cfg.num_hidden_layers as usize;
        let hidden = model.cfg.hidden_size as usize;
        let mut cache = cache(model);
        model
            .forward_gpu_last_logits(
                &tokens(0, 20),
                &positions_for_range(0, 20),
                &mut cache,
                SlotId(0),
            )
            .expect("history prefill");
        let mut session = DFlashCaptureSession::new((0..layers).collect(), 1, hidden, false);
        let (_logits, residual) = model
            .forward_gpu_with_hidden_dflash(
                &tokens(20, 1),
                &positions_for_range(20, 1),
                &mut cache,
                Some(&mut session),
                SlotId(0),
            )
            .expect("captured one-row forward");
        let returned = super::gpu_full_attn::download_f32(&residual).expect("returned residual");
        let last = &session.hidden_output[(layers - 1) * hidden..layers * hidden];
        assert!(
            bits(last) == bits(&returned[..hidden]),
            "the last layer's one-row capture is not the returned residual"
        );
        assert!(
            last.iter().any(|value| *value != 0.0),
            "the one-row capture is all zero"
        );
    });
}
