//! `ImatrixCalibrator` — `Calibrator` trait wrapper around the
//! [`super::imatrix::ImatrixCollector`] algorithm (ADR-014 P7 iter-8).
//!
//! ## Why a separate file from `imatrix.rs`
//!
//! [`super::imatrix`] holds the **algorithm** (`Stats`, `ImatrixCollector`,
//! `accumulate_dense`, `accumulate_moe`, `record_chunk`, `finalise`,
//! `save_imatrix_*`, `load_imatrix_*`) — the byte-identical pure-Rust
//! port of llama.cpp's importance-matrix code.
//!
//! This file holds the **orchestration** — drives an
//! [`ActivationCapture`] over a [`CalibrationCorpus`], feeds each
//! captured layer into the collector, and emits the resulting
//! [`CalibrationData::ImatrixWithStats`] for the
//! [`crate::quantize::Quantizer`] codebook search to consume.
//!
//! The split keeps the algorithm reusable for callers that want to
//! pre-compute imatrix sidecars without going through the trait
//! (e.g. the GGUF-loading bridge in
//! [`super::calibrator::CalibrationData::from_imatrix_gguf`]) while
//! the trait wrapper is the "live" path used by `cmd_convert` when no
//! pre-computed sidecar is provided.
//!
//! ## Forward-pass driver
//!
//! Constructed with a `Box<dyn ActivationCapture>` injected at build
//! time. For qwen35 / qwen35moe, that's
//! `RealActivationCapture::from_model(...)`. For Gemma-4 (P9), that's
//! the same trait implemented by an arch-agnostic CPU forward (out of
//! scope for this iter — Gemma-4 lands in P9 once its
//! `ActivationCapture` impl exists; calling `calibrate(...)` for an
//! arch without an `ActivationCapture` impl is the caller's
//! responsibility).
//!
//! ## Per-layer tensor naming
//!
//! Each captured layer's activation is fed into the collector under a
//! synthetic per-layer tensor name `blk.<l>.layer_input`. This is
//! deliberately **not** a real GGUF tensor name — the calibrator's
//! output is the per-layer importance vector keyed by layer index, not
//! the per-tensor importance vector keyed by Linear weight name (which
//! the legacy llama.cpp imatrix sidecar carries). The downstream
//! consumer (codebook search) maps `blk.<l>.layer_input` →
//! per-Linear-tensor weights inside that layer when applying the
//! per-row imatrix-weighted MSE.

use tracing::{debug, info};

use super::calibrator::{
    CalibrationCorpus, CalibrationData, CalibrationError, Calibrator,
};
use super::imatrix::ImatrixCollector;
use crate::inference::models::qwen35::activation_capture::ActivationCapture;
use crate::ir::lazy::LazyTensorMap;
use crate::ir::ModelMetadata;
use crate::progress::ProgressReporter;

/// Synthetic tensor-name prefix used by the calibrator when feeding
/// captured layer-input activations into the [`ImatrixCollector`]. The
/// downstream codebook search maps `blk.<l>.layer_input` → per-Linear
/// weight names within the same block index.
///
/// `#[allow(dead_code)]` until P8 wires `--quant imatrix-q4_k_m`
/// through `select_calibrator` → `ImatrixCalibrator::new` → live
/// `calibrate(...)` invocation; the constants are public-API forward
/// surface for that consumer.
#[allow(dead_code)]
pub const LAYER_INPUT_TENSOR_PREFIX: &str = "blk";
/// Synthetic suffix completing the tensor name `blk.<l>.layer_input`.
#[allow(dead_code)]
pub const LAYER_INPUT_TENSOR_SUFFIX: &str = "layer_input";

/// `ImatrixCalibrator` — drives a forward pass over a
/// [`CalibrationCorpus`] and accumulates per-layer activation
/// importance via [`ImatrixCollector`]. Consumes any
/// [`ActivationCapture`] impl, so the same calibrator works for
/// qwen35 / qwen35moe (real capture) and for tests
/// ([`crate::inference::models::qwen35::activation_capture::MockActivationCapture`]).
///
/// `#[allow(dead_code)]` on the struct + impl block until P8 wires
/// `--quant imatrix-q4_k_m` through `select_calibrator` → live
/// `calibrate(...)` invocation. Constructor and helpers are exercised
/// by the unit-test suite below.
#[allow(dead_code)]
pub struct ImatrixCalibrator {
    capture: Box<dyn ActivationCapture + Send + Sync>,
    num_layers: u32,
    hidden_size: u32,
}

impl std::fmt::Debug for ImatrixCalibrator {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ImatrixCalibrator")
            .field("num_layers", &self.num_layers)
            .field("hidden_size", &self.hidden_size)
            .field("capture", &"<dyn ActivationCapture>")
            .finish()
    }
}

#[allow(dead_code)]
impl ImatrixCalibrator {
    /// Construct an `ImatrixCalibrator` driven by the given
    /// [`ActivationCapture`].
    ///
    /// `num_layers` and `hidden_size` are validated at calibrate-time
    /// against the captured activations to guard against
    /// model/metadata drift; passing wrong values surfaces a clear
    /// shape-mismatch error rather than silently corrupting the
    /// accumulator.
    pub fn new(
        capture: Box<dyn ActivationCapture + Send + Sync>,
        num_layers: u32,
        hidden_size: u32,
    ) -> Self {
        Self {
            capture,
            num_layers,
            hidden_size,
        }
    }

    /// Build the synthetic per-layer tensor name used by the collector.
    fn layer_tensor_name(layer_idx: usize) -> String {
        format!(
            "{}.{}.{}",
            LAYER_INPUT_TENSOR_PREFIX, layer_idx, LAYER_INPUT_TENSOR_SUFFIX
        )
    }
}

impl Calibrator for ImatrixCalibrator {
    fn name(&self) -> &'static str {
        "imatrix"
    }

    fn requires_forward_pass(&self) -> bool {
        true
    }

    fn calibrate(
        &mut self,
        _model: &LazyTensorMap,
        meta: &ModelMetadata,
        corpus: &CalibrationCorpus,
        progress: &ProgressReporter,
    ) -> Result<CalibrationData, CalibrationError> {
        if corpus.is_empty() {
            return Err(CalibrationError::EmptyCorpus);
        }

        // Sanity-check metadata vs constructor params. If they
        // disagree, the per-layer accumulator can't size itself
        // consistently, so refuse rather than silently corrupting.
        if meta.num_layers != self.num_layers {
            return Err(CalibrationError::Other {
                message: format!(
                    "imatrix calibrator: num_layers mismatch (constructor={}, \
                     metadata={}); update the calibrator or the metadata to agree",
                    self.num_layers, meta.num_layers
                ),
            });
        }
        if (meta.hidden_size as u32) != self.hidden_size {
            return Err(CalibrationError::Other {
                message: format!(
                    "imatrix calibrator: hidden_size mismatch (constructor={}, \
                     metadata={}); update the calibrator or the metadata to agree",
                    self.hidden_size, meta.hidden_size
                ),
            });
        }

        info!(
            arch = %meta.architecture,
            num_layers = self.num_layers,
            hidden_size = self.hidden_size,
            n_chunks = corpus.n_chunks(),
            total_tokens = corpus.total_tokens(),
            corpus = %corpus.name,
            "imatrix calibration starting"
        );

        let pb = progress.bar(corpus.n_chunks() as u64, "Imatrix calibration");
        let mut collector = ImatrixCollector::new();

        for (chunk_idx, chunk) in corpus.chunks.iter().enumerate() {
            if chunk.is_empty() {
                pb.inc(1);
                continue;
            }

            let activations =
                self.capture
                    .run_calibration_prompt(chunk)
                    .map_err(|e| CalibrationError::Other {
                        message: format!(
                            "imatrix forward pass failed at chunk {chunk_idx}: {e}"
                        ),
                    })?;

            activations
                .validate()
                .map_err(|e| CalibrationError::Other {
                    message: format!(
                        "imatrix capture: LayerActivations shape invariant violated \
                         at chunk {chunk_idx}: {e}"
                    ),
                })?;

            if activations.num_layers != self.num_layers {
                return Err(CalibrationError::Other {
                    message: format!(
                        "imatrix capture returned {} layers; calibrator expected {} \
                         (chunk {chunk_idx})",
                        activations.num_layers, self.num_layers
                    ),
                });
            }
            if activations.hidden_size != self.hidden_size {
                return Err(CalibrationError::Other {
                    message: format!(
                        "imatrix capture returned hidden_size={}; calibrator expected \
                         {} (chunk {chunk_idx})",
                        activations.hidden_size, self.hidden_size
                    ),
                });
            }

            let seq_len = activations.seq_len as usize;
            let hidden = activations.hidden_size as usize;

            for (layer_idx, layer_acts) in activations.layer_inputs.iter().enumerate() {
                let tensor_name = Self::layer_tensor_name(layer_idx);
                collector.accumulate_dense(
                    &tensor_name,
                    layer_acts,
                    seq_len,
                    hidden,
                )?;
            }

            collector.record_chunk();
            debug!(
                chunk_idx = chunk_idx,
                tokens = chunk.len(),
                tensors_so_far = collector.len(),
                "imatrix chunk recorded"
            );
            pb.inc(1);
        }

        info!(
            chunks = collector.chunks(),
            tensors = collector.len(),
            "imatrix calibration complete"
        );

        let data = CalibrationData::from_imatrix_collector(&collector);
        // Defensive: collector.len() should match data.len(). If not,
        // a future change to from_imatrix_collector dropped tensors
        // silently — surface that loudly.
        if data.len() != collector.len() {
            return Err(CalibrationError::Other {
                message: format!(
                    "imatrix bridge inconsistency: collector tracked {} tensors \
                     but CalibrationData carries {}",
                    collector.len(),
                    data.len()
                ),
            });
        }

        Ok(data)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    use crate::calibrate::imatrix::ImatrixCollector;
    use crate::inference::models::qwen35::activation_capture::MockActivationCapture;

    fn dummy_metadata(num_layers: u32, hidden_size: u32) -> ModelMetadata {
        ModelMetadata {
            architecture: "TestArch".into(),
            model_type: "test".into(),
            param_count: 0,
            hidden_size: hidden_size as u64,
            num_layers,
            layer_types: (0..num_layers).map(|_| "attention".into()).collect(),
            num_attention_heads: 1,
            num_kv_heads: Some(1),
            vocab_size: 256,
            dtype: "float16".into(),
            shard_count: 1,
            num_experts: None,
            top_k_experts: None,
            intermediate_size: None,
            raw_config: serde_json::Value::Null,
            explicit_layer_types: None,
            full_attention_interval: None,
            attn_output_gate: None,
            head_dim: None,
            partial_rotary_factor: None,
            rope_parameters: None,
            linear_conv_kernel_dim: None,
            linear_key_head_dim: None,
            linear_num_key_heads: None,
            linear_value_head_dim: None,
            linear_num_value_heads: None,
            mamba_ssm_dtype: None,
            moe_intermediate_size: None,
            shared_expert_intermediate_size: None,
            mtp_num_hidden_layers: None,
            mtp_use_dedicated_embeddings: None,
            output_router_logits: None,
            router_aux_loss_coef: None,
        }
    }

    fn make_calibrator(num_layers: u32, hidden_size: u32) -> ImatrixCalibrator {
        ImatrixCalibrator::new(
            Box::new(MockActivationCapture::new(num_layers, hidden_size)),
            num_layers,
            hidden_size,
        )
    }

    #[test]
    fn imatrix_calibrator_name_and_requires_forward_pass() {
        let calib = make_calibrator(2, 8);
        assert_eq!(calib.name(), "imatrix");
        assert!(calib.requires_forward_pass());
    }

    #[test]
    fn imatrix_calibrator_empty_corpus_errors() {
        let mut calib = make_calibrator(2, 8);
        let lazy_map = LazyTensorMap::new();
        let meta = dummy_metadata(2, 8);
        let progress = ProgressReporter::new();

        let corpus = CalibrationCorpus {
            chunks: vec![],
            name: "empty".into(),
        };
        let err = calib
            .calibrate(&lazy_map, &meta, &corpus, &progress)
            .unwrap_err();
        match err {
            CalibrationError::EmptyCorpus => {}
            other => panic!("expected EmptyCorpus, got {other:?}"),
        }

        // Also: a corpus with empty inner chunks is "empty" via
        // CalibrationCorpus::is_empty.
        let corpus2 = CalibrationCorpus {
            chunks: vec![vec![], vec![]],
            name: "all-empty".into(),
        };
        let err2 = calib
            .calibrate(&lazy_map, &meta, &corpus2, &progress)
            .unwrap_err();
        assert!(matches!(err2, CalibrationError::EmptyCorpus));
    }

    /// Synthetic Gemma-4-shaped round trip via `MockActivationCapture`
    /// (Gemma-4's real ActivationCapture lands in P9; this test
    /// validates the calibrator's contract using the mock, which is
    /// the exact pattern Gemma-4 will follow once its capture exists).
    #[test]
    fn imatrix_calibrator_synthetic_gemma4_round_trip() {
        let num_layers = 4u32;
        let hidden_size = 8u32;
        let mut calib = make_calibrator(num_layers, hidden_size);
        let lazy_map = LazyTensorMap::new();
        let meta = dummy_metadata(num_layers, hidden_size);
        let progress = ProgressReporter::new();

        // 2 chunks of 16 tokens each.
        let corpus = CalibrationCorpus {
            chunks: vec![(0..16u32).collect(), (16..32u32).collect()],
            name: "synthetic-gemma4".into(),
        };
        assert_eq!(corpus.n_chunks(), 2);

        let data = calib
            .calibrate(&lazy_map, &meta, &corpus, &progress)
            .expect("calibration succeeds");

        // Returns ImatrixWithStats.
        let stats_map = match data {
            CalibrationData::ImatrixWithStats(m) => m,
            other => panic!("expected ImatrixWithStats, got {other:?}"),
        };

        // One entry per layer.
        assert_eq!(stats_map.len(), num_layers as usize);

        for layer_idx in 0..num_layers as usize {
            let name = ImatrixCalibrator::layer_tensor_name(layer_idx);
            let stat = stats_map
                .get(&name)
                .unwrap_or_else(|| panic!("missing stat entry for {name}"));

            // Dense layer: counts.len() == 1, values.len() == hidden_size.
            assert_eq!(stat.counts.len(), 1, "dense expects single count slot");
            assert_eq!(
                stat.values.len(),
                hidden_size as usize,
                "values vec must match hidden_size"
            );
            // Token-count accumulation: 2 chunks → counts[0] == 2.
            assert_eq!(
                stat.counts[0], 2,
                "expected 2 chunks recorded, got {}",
                stat.counts[0]
            );
            // Mock activations are non-zero by construction (see
            // MockActivationCapture's `base = tokens[t]*0.001 + l*0.01 + j*0.0001`),
            // so the squared-activation accumulator must be strictly positive.
            assert!(
                stat.values.iter().any(|&v| v > 0.0),
                "expected non-zero importance for layer {layer_idx}; got {:?}",
                stat.values
            );
        }
    }

    /// Object-safe trait usage — required for the dispatch helper in
    /// `select_calibrator(...)` returning `Box<dyn Calibrator>`.
    #[test]
    fn imatrix_calibrator_object_safe() {
        let calib = make_calibrator(2, 8);
        let _: Box<dyn Calibrator> = Box::new(calib);
    }

    /// Send + Sync — required by the trait bound for future rayon
    /// parallelism.
    #[test]
    fn imatrix_calibrator_is_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<ImatrixCalibrator>();
    }

    /// Number of chunks recorded in the collector matches the number
    /// of non-empty chunks in the corpus. Verified by reconstructing
    /// the same accumulation manually and comparing to the calibrator's
    /// output.
    #[test]
    fn imatrix_calibrator_corpus_token_count_matches_hand_computed() {
        let num_layers = 2u32;
        let hidden_size = 4u32;
        let mut calib = make_calibrator(num_layers, hidden_size);
        let lazy_map = LazyTensorMap::new();
        let meta = dummy_metadata(num_layers, hidden_size);
        let progress = ProgressReporter::new();

        let chunks = vec![vec![1u32, 2, 3], vec![4u32, 5]];
        let corpus = CalibrationCorpus {
            chunks: chunks.clone(),
            name: "hand".into(),
        };

        let data = calib
            .calibrate(&lazy_map, &meta, &corpus, &progress)
            .unwrap();
        let stats_map = match data {
            CalibrationData::ImatrixWithStats(m) => m,
            _ => unreachable!(),
        };

        // Hand-compute the expected per-layer values via the mock
        // formula: input[l][t,j] = tokens[t]*0.001 + l*0.01 + j*0.0001.
        // The collector accumulates per-batch summed-then-mean, so the
        // expected value at layer l, col j after processing all chunks is:
        //   sum over chunks c of mean_over_tokens_t( input[l][t,j]² )
        let mut expected: HashMap<String, Vec<f32>> = HashMap::new();
        for l in 0..num_layers as usize {
            let mut values = vec![0.0_f32; hidden_size as usize];
            for chunk in &chunks {
                let n_tokens = chunk.len() as f32;
                for (j, slot) in values.iter_mut().enumerate() {
                    let mut sumf = 0.0_f32;
                    for &tok in chunk {
                        let v = (tok as f32) * 0.001
                            + (l as f32) * 0.01
                            + (j as f32) * 0.0001;
                        sumf += v * v;
                    }
                    *slot += sumf / n_tokens;
                }
            }
            expected.insert(ImatrixCalibrator::layer_tensor_name(l), values);
        }

        for (k, exp) in &expected {
            let got = stats_map.get(k).expect("layer present");
            assert_eq!(got.values.len(), exp.len());
            for (i, (a, b)) in got.values.iter().zip(exp.iter()).enumerate() {
                assert!(
                    (a - b).abs() <= 1e-5,
                    "layer {k} col {i}: got {a}, expected {b} (diff {})",
                    (a - b).abs()
                );
            }
        }
    }

    /// Construction validates capture parameters at calibrate time:
    /// passing metadata that disagrees with the constructor's
    /// num_layers surfaces a typed error — no silent truncation.
    #[test]
    fn imatrix_calibrator_metadata_layer_mismatch_errors() {
        let mut calib = make_calibrator(4, 8);
        let lazy_map = LazyTensorMap::new();
        let bad_meta = dummy_metadata(3, 8); // disagrees
        let progress = ProgressReporter::new();
        let corpus = CalibrationCorpus {
            chunks: vec![vec![1u32, 2]],
            name: "x".into(),
        };
        let err = calib
            .calibrate(&lazy_map, &bad_meta, &corpus, &progress)
            .unwrap_err();
        match err {
            CalibrationError::Other { message } => {
                assert!(
                    message.contains("num_layers"),
                    "error must cite num_layers, got: {message}"
                );
            }
            other => panic!("expected Other(num_layers ...), got {other:?}"),
        }
    }

    /// Bridge consistency: the calibrator's output entry count equals
    /// the underlying collector's tracked-tensor count. Validates that
    /// `from_imatrix_collector` doesn't silently drop entries.
    #[test]
    fn imatrix_calibrator_bridge_no_drop() {
        let num_layers = 3u32;
        let hidden_size = 6u32;
        let mut calib = make_calibrator(num_layers, hidden_size);
        let lazy_map = LazyTensorMap::new();
        let meta = dummy_metadata(num_layers, hidden_size);
        let progress = ProgressReporter::new();

        let corpus = CalibrationCorpus {
            chunks: vec![vec![10u32, 20, 30]],
            name: "bridge".into(),
        };
        let data = calib
            .calibrate(&lazy_map, &meta, &corpus, &progress)
            .unwrap();

        match data {
            CalibrationData::ImatrixWithStats(m) => {
                assert_eq!(m.len(), num_layers as usize);
            }
            other => panic!("expected ImatrixWithStats, got {other:?}"),
        }
    }

    /// The collector this calibrator builds is the same algorithm used
    /// by `accumulate_dense` directly — so a manual call to the
    /// collector produces an equivalent shape. Anchors the
    /// algorithm-vs-orchestration split: the trait wrapper is purely
    /// orchestration; correctness lives in [`ImatrixCollector`].
    #[test]
    fn imatrix_calibrator_uses_underlying_collector_shape() {
        let mut col = ImatrixCollector::new();
        col.accumulate_dense("blk.0.layer_input", &[1.0, 2.0, 3.0, 4.0], 4, 1)
            .unwrap();
        col.record_chunk();
        let stats = col.stats().get("blk.0.layer_input").unwrap();
        assert_eq!(stats.counts.len(), 1);
        // Same shape as what ImatrixCalibrator would produce for a
        // single dense-layer chunk under the same (n_tokens=4, row=1)
        // shape.
    }
}
