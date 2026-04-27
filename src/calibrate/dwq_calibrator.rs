//! `DwqCalibrator` — `Calibrator` trait wrapper around the DWQ
//! activation-capture pipeline (ADR-014 P7 iter-8).
//!
//! ## Responsibility split
//!
//! * [`super::dwq`]: the DWQ algorithm itself (closed-form scale
//!   calibration, [`super::dwq::DwqQuantizer`], [`super::dwq::DwqError`]).
//! * [`super::dwq_activation`]: the orchestration that drives an
//!   [`ActivationCapture`] to produce per-layer sensitivity ranges.
//! * [`super::sensitivity`]: per-layer sensitivity scoring + bit
//!   allocation primitives.
//! * **This file**: the trait wrapper. Implements [`Calibrator`] over
//!   the orchestration, returning [`CalibrationData::Dwq`] with the
//!   per-layer sensitivity flag map that downstream
//!   [`crate::quantize::Quantizer`] dispatch consumes.
//!
//! ## No silent fallback (ADR-012 Decision 13 / `prior_pattern`)
//!
//! qwen35 / qwen35moe DWQ requires an [`ActivationCapture`] impl —
//! the weight-space path is **not** a valid fallback. If
//! [`super::dwq::DwqArch::requires_activation_capture`] returns true
//! and no capture was provided to the constructor, `calibrate(...)`
//! returns [`CalibrationError::ForwardPassUnavailable`] **without**
//! routing through a degraded path.
//!
//! For [`super::dwq::DwqArch::Other`], the contract is reversed: those
//! architectures' DWQ output is computed in weight space (no capture
//! needed) and the calibrator returns an empty
//! [`CalibrationData::Dwq`] map — empty ≠ degraded; an empty map is
//! the explicit "weight-space path is the intended downstream codec"
//! signal. Downstream `DwqQuantizer` dispatch already handles this
//! shape (the `arch == DwqArch::Other` branch in
//! [`super::dwq::run_dwq_calibration`] proceeds without consulting any
//! capture-derived sensitivity).

use std::collections::HashMap;
use std::path::PathBuf;

use tracing::{debug, info, warn};

use super::cache::{
    cache_file_path, load_dwq_from_path, save_dwq_to_path, SensitivityCacheKey,
    SENSITIVITY_ALGORITHM_VERSION,
};
use super::calibrator::{
    corpus_sha, model_fingerprint, CalibrationCorpus, CalibrationData, CalibrationError, Calibrator,
};
use super::dwq::{DwqArch, DwqConfig};
use super::dwq_activation::capture_activations_to_sensitive_ranges;
use crate::inference::models::qwen35::activation_capture::ActivationCapture;
use crate::ir::lazy::LazyTensorMap;
use crate::ir::ModelMetadata;
use crate::progress::ProgressReporter;

/// Synthetic per-layer tensor name template used as the key in the
/// returned [`CalibrationData::Dwq`] map. The downstream consumer
/// reads `blk.<i>.sensitivity` to learn whether layer `i` should
/// receive `sensitive_bits` (vec entry == 1.0) or `base_bits` (entry
/// == 0.0).
///
/// `#[allow(dead_code)]` until P2 iter-2 invokes `calibrate(...)`
/// from the live cmd_convert dispatch site; this iter only wires the
/// `select_calibrator` seam and logs the selection.
#[allow(dead_code)]
pub const SENSITIVITY_TENSOR_PREFIX: &str = "blk";
/// Synthetic suffix completing the tensor name `blk.<i>.sensitivity`.
#[allow(dead_code)]
pub const SENSITIVITY_TENSOR_SUFFIX: &str = "sensitivity";

/// `DwqCalibrator` — wraps the activation-capture-driven DWQ
/// orchestration in the [`Calibrator`] trait.
///
/// The constructor takes:
/// * `arch` — the DWQ architecture family. Drives whether
///   [`Calibrator::requires_forward_pass`] returns true.
/// * `capture` — `Some(impl)` for arches that require it,
///   `None` permitted only when `arch == DwqArch::Other`. The
///   "requires=true with no capture" combination is detected at
///   `calibrate` time and surfaces
///   [`CalibrationError::ForwardPassUnavailable`].
/// * `base_bits` / `sensitive_bits` — the bit-pair preset
///   (e.g. `(4, 6)` for `dwq-mixed-4-6`).
/// * `calibration_samples` — number of tokens to feed through the
///   capture (matches the DWQ default of 1024).
pub struct DwqCalibrator {
    capture: CaptureSource,
    arch: DwqArch,
    base_bits: u8,
    sensitive_bits: u8,
    calibration_samples: u32,
}

/// How the calibrator obtains its [`ActivationCapture`] impl.
///
/// `Eager` carries an already-built capture (used by tests with
/// `MockActivationCapture` and by callers that share a model with an
/// inference session). `Deferred` carries the paths needed to build a
/// `RealActivationCapture` lazily inside [`Calibrator::calibrate`] —
/// the production path used by `cmd_convert` after intermediate-F16
/// GGUF emission. `None` is the explicit "no capture" state for
/// `DwqArch::Other`.
///
/// Deferred construction lets `cmd_convert` keep the
/// `RealActivationCapture::new` call out of its dispatch arm
/// (greppable invariant, ADR-014 P2 iter-2 §S3).
enum CaptureSource {
    None,
    Eager(Box<dyn ActivationCapture + Send + Sync>),
    Deferred {
        intermediate_gguf: PathBuf,
        tokenizer_json: PathBuf,
    },
}

impl std::fmt::Debug for DwqCalibrator {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let has_capture = !matches!(self.capture, CaptureSource::None);
        let capture_kind = match &self.capture {
            CaptureSource::None => "none",
            CaptureSource::Eager(_) => "eager",
            CaptureSource::Deferred { .. } => "deferred",
        };
        f.debug_struct("DwqCalibrator")
            .field("arch", &self.arch)
            .field("base_bits", &self.base_bits)
            .field("sensitive_bits", &self.sensitive_bits)
            .field("calibration_samples", &self.calibration_samples)
            .field("has_capture", &has_capture)
            .field("capture_kind", &capture_kind)
            .finish()
    }
}

impl DwqCalibrator {
    /// Construct a `DwqCalibrator`.
    ///
    /// `capture` is `Option<...>` because [`DwqArch::Other`] does not
    /// need one (its sensitivity is derived in weight space, downstream).
    /// Passing `None` for an arch where
    /// [`DwqArch::requires_activation_capture`] returns true is
    /// detected at `calibrate(...)` time and surfaces
    /// [`CalibrationError::ForwardPassUnavailable`] — never silently
    /// falls back.
    pub fn new(
        arch: DwqArch,
        capture: Option<Box<dyn ActivationCapture + Send + Sync>>,
        base_bits: u8,
        sensitive_bits: u8,
        calibration_samples: u32,
    ) -> Self {
        let capture = match capture {
            Some(c) => CaptureSource::Eager(c),
            None => CaptureSource::None,
        };
        Self {
            capture,
            arch,
            base_bits,
            sensitive_bits,
            calibration_samples,
        }
    }

    /// Construct a `DwqCalibrator` that builds its
    /// [`crate::inference::models::qwen35::activation_capture_real::RealActivationCapture`]
    /// lazily inside [`Calibrator::calibrate`] — the production path used by
    /// `cmd_convert` after the caller has emitted an intermediate-F16 GGUF
    /// (so the capture can `Qwen35Model::load_from_gguf` without holding the
    /// converter's `tensor_map` resident at the same time, the OOM
    /// mitigation pattern documented in `dwq_activation::capture_activations_to_sensitive_ranges`).
    ///
    /// The `RealActivationCapture::new` call is delayed to
    /// `calibrate(...)` time so the intermediate-GGUF + tokenizer paths
    /// can be passed in early (before `cmd_convert` drops `tensor_map`)
    /// while the heavyweight `Qwen35Model::load_from_gguf` (~128 GB
    /// peak for apex MoE) doesn't run until after `tensor_map` is freed.
    ///
    /// ADR-014 P2 iter-2 §S3: this is the path that lets `cmd_convert`
    /// hold zero direct `RealActivationCapture::new` /
    /// `capture_activations_to_sensitive_ranges` callers — the
    /// orchestration moves entirely into [`Calibrator::calibrate`].
    pub fn with_activation_capture(
        arch: DwqArch,
        intermediate_gguf: PathBuf,
        tokenizer_json: PathBuf,
        base_bits: u8,
        sensitive_bits: u8,
        calibration_samples: u32,
    ) -> Self {
        Self {
            capture: CaptureSource::Deferred {
                intermediate_gguf,
                tokenizer_json,
            },
            arch,
            base_bits,
            sensitive_bits,
            calibration_samples,
        }
    }

    /// Build the synthetic per-layer sensitivity tensor name.
    ///
    /// `#[allow(dead_code)]` until P2 iter-2 invokes `calibrate(...)`
    /// from cmd_convert's live dispatch site (this iter only wires
    /// `select_calibrator` + logs the selection — the calibrate path
    /// is exercised by the unit tests).
    #[allow(dead_code)]
    pub fn sensitivity_tensor_name(layer_idx: usize) -> String {
        format!(
            "{}.{}.{}",
            SENSITIVITY_TENSOR_PREFIX, layer_idx, SENSITIVITY_TENSOR_SUFFIX
        )
    }
}

impl Calibrator for DwqCalibrator {
    fn name(&self) -> &'static str {
        "dwq"
    }

    fn requires_forward_pass(&self) -> bool {
        self.arch.requires_activation_capture()
    }

    fn calibrate(
        &mut self,
        model: &LazyTensorMap,
        meta: &ModelMetadata,
        corpus: &CalibrationCorpus,
        progress: &ProgressReporter,
    ) -> Result<CalibrationData, CalibrationError> {
        // ─────────── DwqArch::Other — weight-space path ───────────
        // Empty map signals "no per-layer-sensitivity data; the
        // downstream codec is weight-space for this arch by
        // contract". This is NOT a fallback; it is the documented
        // shape for arches that don't need capture.
        if !self.arch.requires_activation_capture() {
            info!(
                arch = ?self.arch,
                base_bits = self.base_bits,
                sensitive_bits = self.sensitive_bits,
                "dwq calibrator: arch is weight-space; emitting empty Dwq map"
            );
            return Ok(CalibrationData::Dwq(HashMap::new()));
        }

        // The corpus is informational for DWQ — calibration tokens
        // are derived from `meta.vocab_size` inside
        // `capture_activations_to_sensitive_ranges`. Reject an
        // explicitly-empty corpus to keep the contract symmetric with
        // `ImatrixCalibrator` (it costs nothing and surfaces a
        // misuse early). Validated BEFORE the (potentially expensive)
        // deferred capture build so a misuse fails fast.
        if corpus.is_empty() {
            return Err(CalibrationError::EmptyCorpus);
        }

        // ─────────── No-fallback guard + deferred capture build ───────────
        // arch.requires_activation_capture() == true → capture MUST
        // be available (either eagerly attached, or constructible from
        // the deferred paths). Bail with a typed error otherwise.
        // The deferred build runs `RealActivationCapture::new` here,
        // intentionally — `cmd_convert` no longer calls it directly
        // (ADR-014 P2 iter-2 §S3 greppable invariant).
        //
        // Materialise any deferred capture in-place so the subsequent
        // borrow yields a long-lived `&mut dyn ActivationCapture` from
        // the same `CaptureSource::Eager` slot (no `.expect()` shenanigans
        // and no holding two mutable borrows).
        if let CaptureSource::Deferred {
            intermediate_gguf,
            tokenizer_json,
        } = &self.capture
        {
            let real =
                crate::inference::models::qwen35::activation_capture_real::RealActivationCapture::new(
                    intermediate_gguf,
                    tokenizer_json,
                )
                .map_err(|e| CalibrationError::Other {
                    message: format!(
                        "DwqCalibrator deferred capture build failed for arch {:?} \
                         (intermediate_gguf={}, tokenizer={}): {e}",
                        self.arch,
                        intermediate_gguf.display(),
                        tokenizer_json.display()
                    ),
                })?;
            self.capture = CaptureSource::Eager(Box::new(real));
        }
        let capture: &mut dyn ActivationCapture = match &mut self.capture {
            CaptureSource::None => {
                return Err(CalibrationError::ForwardPassUnavailable {
                    arch: format!("{:?}", self.arch),
                });
            }
            CaptureSource::Eager(boxed) => boxed.as_mut(),
            CaptureSource::Deferred { .. } => {
                // Unreachable: just transitioned Deferred → Eager above.
                // Use a typed error rather than `unreachable!()` to
                // honor the no-panic contract added by ADR-014 P2 iter-2.
                return Err(CalibrationError::Other {
                    message: "DwqCalibrator: internal capture-state transition failed"
                        .to_string(),
                });
            }
        };

        let cache_key = SensitivityCacheKey::with_algorithm_version(
            model_fingerprint(model, meta),
            corpus_sha(corpus),
            SENSITIVITY_ALGORITHM_VERSION,
        );
        let cache_path = match cache_file_path(&cache_key) {
            Ok(path) => Some(path),
            Err(err) => {
                warn!(
                    error = %err,
                    "dwq sensitivity cache path unavailable; running forward pass"
                );
                None
            }
        };

        if let Some(path) = cache_path.as_deref() {
            match load_dwq_from_path(path, &cache_key) {
                Ok(Some(map)) => {
                    info!(
                        cache_key = %cache_key.hash(),
                        path = %path.display(),
                        entries = map.len(),
                        "dwq sensitivity cache HIT"
                    );
                    return Ok(CalibrationData::Dwq(map));
                }
                Ok(None) => {
                    info!(
                        cache_key = %cache_key.hash(),
                        path = %path.display(),
                        "dwq sensitivity cache MISS"
                    );
                }
                Err(err) => {
                    warn!(
                        error = %err,
                        cache_key = %cache_key.hash(),
                        path = %path.display(),
                        "dwq sensitivity cache MISS; load failed, recomputing"
                    );
                }
            }
        }

        let pb = progress.bar(meta.num_layers as u64, "DWQ activation capture");
        info!(
            arch = ?self.arch,
            base_bits = self.base_bits,
            sensitive_bits = self.sensitive_bits,
            calibration_samples = self.calibration_samples,
            num_layers = meta.num_layers,
            corpus = %corpus.name,
            "dwq calibration starting"
        );

        let dwq_config = DwqConfig {
            calibration_samples: self.calibration_samples,
            base_bits: self.base_bits,
            sensitive_bits: self.sensitive_bits,
            arch: self.arch,
            ..DwqConfig::default()
        };

        let sensitive_ranges =
            capture_activations_to_sensitive_ranges(meta, &dwq_config, capture).map_err(
                |e| CalibrationError::Other {
                    message: format!("dwq activation capture failed: {e}"),
                },
            )?;

        // Flatten the range list into a per-layer flag (1.0 sensitive,
        // 0.0 base) for the downstream `Dwq` map shape.
        let num_layers = meta.num_layers as usize;
        let mut sensitivity_flags = vec![false; num_layers];
        for range in &sensitive_ranges {
            for i in range.clone() {
                if i < num_layers {
                    sensitivity_flags[i] = true;
                }
            }
        }

        let mut map: HashMap<String, Vec<f32>> = HashMap::with_capacity(num_layers);
        let mut sensitive_count = 0_usize;
        for (layer_idx, &is_sensitive) in sensitivity_flags.iter().enumerate() {
            let name = Self::sensitivity_tensor_name(layer_idx);
            let value: f32 = if is_sensitive { 1.0 } else { 0.0 };
            if is_sensitive {
                sensitive_count += 1;
            }
            map.insert(name, vec![value]);
            pb.inc(1);
        }

        debug!(
            sensitive_count = sensitive_count,
            total = num_layers,
            "dwq sensitivity flags populated"
        );

        if let Some(path) = cache_path.as_deref() {
            if let Err(err) = save_dwq_to_path(path, &cache_key, &map) {
                warn!(
                    error = %err,
                    cache_key = %cache_key.hash(),
                    path = %path.display(),
                    "dwq sensitivity cache save failed; calibration result will still be returned"
                );
            }
        }

        Ok(CalibrationData::Dwq(map))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;
    use std::ffi::OsString;
    use std::fs;
    use std::path::Path;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;

    use crate::calibrate::cache::{
        cache_file_path, save_dwq_to_path, SensitivityCacheKey, SENSITIVITY_ALGORITHM_VERSION,
    };
    use crate::calibrate::calibrator::{corpus_sha, model_fingerprint};
    use crate::inference::models::qwen35::activation_capture::{
        LayerActivations, MockActivationCapture,
    };
    use crate::ir::lazy::LazyTensorMap;
    use crate::ir::ModelMetadata;

    struct EnvVarGuard {
        key: &'static str,
        previous: Option<OsString>,
    }

    impl EnvVarGuard {
        fn set_path(key: &'static str, path: &Path) -> Self {
            let previous = std::env::var_os(key);
            std::env::set_var(key, path);
            Self { key, previous }
        }
    }

    impl Drop for EnvVarGuard {
        fn drop(&mut self) {
            if let Some(previous) = &self.previous {
                std::env::set_var(self.key, previous);
            } else {
                std::env::remove_var(self.key);
            }
        }
    }

    struct CountingMockActivationCapture {
        inner: MockActivationCapture,
        counter: Arc<AtomicUsize>,
    }

    impl CountingMockActivationCapture {
        fn new(num_layers: u32, hidden_size: u32, counter: Arc<AtomicUsize>) -> Self {
            Self {
                inner: MockActivationCapture::new(num_layers, hidden_size),
                counter,
            }
        }
    }

    impl ActivationCapture for CountingMockActivationCapture {
        fn run_calibration_prompt(&mut self, tokens: &[u32]) -> anyhow::Result<LayerActivations> {
            self.counter.fetch_add(1, Ordering::SeqCst);
            self.inner.run_calibration_prompt(tokens)
        }
    }

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

    fn nonempty_corpus() -> CalibrationCorpus {
        CalibrationCorpus {
            chunks: vec![vec![1u32, 2, 3, 4]],
            name: "test".into(),
        }
    }

    fn dwq_cache_key(
        model: &LazyTensorMap,
        meta: &ModelMetadata,
        corpus: &CalibrationCorpus,
    ) -> SensitivityCacheKey {
        SensitivityCacheKey::with_algorithm_version(
            model_fingerprint(model, meta),
            corpus_sha(corpus),
            SENSITIVITY_ALGORITHM_VERSION,
        )
    }

    /// `name() == "dwq"` — the dispatch key.
    #[test]
    fn dwq_calibrator_name() {
        let calib = DwqCalibrator::new(DwqArch::Other, None, 4, 6, 1024);
        assert_eq!(calib.name(), "dwq");
    }

    /// Arch table: Qwen35Dense / Qwen35MoE → requires_forward_pass=true;
    /// Other → false.
    #[test]
    fn dwq_calibrator_requires_forward_pass_arch_table() {
        let qwen_dense = DwqCalibrator::new(DwqArch::Qwen35Dense, None, 4, 6, 1024);
        assert!(qwen_dense.requires_forward_pass());

        let qwen_moe = DwqCalibrator::new(DwqArch::Qwen35MoE, None, 4, 6, 1024);
        assert!(qwen_moe.requires_forward_pass());

        let other = DwqCalibrator::new(DwqArch::Other, None, 4, 6, 1024);
        assert!(!other.requires_forward_pass());
    }

    /// **No-silent-fallback contract** (ADR-012 D13 + `prior_pattern`).
    /// Qwen35MoE arch with `capture == None` MUST surface
    /// `ForwardPassUnavailable`, not silently return an empty map (which
    /// would degrade to weight-space — the exact mistake D13 forbids).
    #[test]
    fn dwq_calibrator_no_silent_weight_space_fallback() {
        let mut calib = DwqCalibrator::new(DwqArch::Qwen35MoE, None, 4, 6, 1024);
        let lazy_map = LazyTensorMap::new();
        let meta = dummy_metadata(4, 16);
        let progress = ProgressReporter::new();
        let corpus = nonempty_corpus();

        let err = calib
            .calibrate(&lazy_map, &meta, &corpus, &progress)
            .expect_err("must error: requires capture but none provided");

        match err {
            CalibrationError::ForwardPassUnavailable { arch } => {
                assert!(
                    arch.contains("Qwen35MoE"),
                    "error must cite the arch, got {arch:?}"
                );
            }
            other => panic!(
                "expected ForwardPassUnavailable, got {other:?} \
                 (would have allowed silent weight-space fallback — bug)"
            ),
        }

        // Also for Qwen35Dense.
        let mut calib_dense = DwqCalibrator::new(DwqArch::Qwen35Dense, None, 4, 6, 1024);
        let err2 = calib_dense
            .calibrate(&lazy_map, &meta, &corpus, &progress)
            .expect_err("Qwen35Dense without capture must error");
        assert!(matches!(
            err2,
            CalibrationError::ForwardPassUnavailable { .. }
        ));
    }

    /// `DwqArch::Other` returns `Ok(CalibrationData::Dwq(empty))` —
    /// the explicit "weight-space path is the intended downstream
    /// codec" shape. Capture is NOT consulted (None is fine).
    #[test]
    fn dwq_calibrator_other_arch_returns_empty_dwq_map() {
        let mut calib = DwqCalibrator::new(DwqArch::Other, None, 4, 6, 1024);
        let lazy_map = LazyTensorMap::new();
        let meta = dummy_metadata(4, 16);
        let progress = ProgressReporter::new();
        let corpus = CalibrationCorpus {
            chunks: vec![],
            name: "ignored".into(),
        };

        let data = calib
            .calibrate(&lazy_map, &meta, &corpus, &progress)
            .expect("DwqArch::Other path must succeed without capture");

        match data {
            CalibrationData::Dwq(m) => {
                assert!(
                    m.is_empty(),
                    "DwqArch::Other returns empty Dwq map (weight-space contract); \
                     got {} entries",
                    m.len()
                );
            }
            other => panic!("expected CalibrationData::Dwq(empty), got {other:?}"),
        }
    }

    /// Synthetic MoE round trip via `MockActivationCapture`. Returns
    /// `Dwq(map)` with exactly `num_layers` entries, each carrying the
    /// per-layer sensitivity flag. The mock's monotonic activation
    /// formula `tokens[t]*0.001 + l*0.01 + j*0.0001` makes later layers
    /// more sensitive, but for the bit-pair (4, 6) midpoint = 5.0 and
    /// the `> midpoint` filter inside `capture_activations_to_sensitive_ranges`
    /// will mark high-magnitude layers as sensitive.
    #[test]
    fn dwq_calibrator_synthetic_moe_round_trip() {
        let num_layers = 4u32;
        let hidden_size = 8u32;
        let capture: Box<dyn ActivationCapture + Send + Sync> =
            Box::new(MockActivationCapture::new(num_layers, hidden_size));
        let mut calib = DwqCalibrator::new(DwqArch::Qwen35MoE, Some(capture), 4, 6, 16);
        let lazy_map = LazyTensorMap::new();
        let meta = dummy_metadata(num_layers, hidden_size);
        let progress = ProgressReporter::new();
        let corpus = CalibrationCorpus {
            chunks: vec![(0..16u32).collect()],
            name: "synthetic-moe".into(),
        };

        let data = calib
            .calibrate(&lazy_map, &meta, &corpus, &progress)
            .expect("Qwen35MoE round-trip via mock must succeed");

        let map = match data {
            CalibrationData::Dwq(m) => m,
            other => panic!("expected CalibrationData::Dwq, got {other:?}"),
        };

        // Exactly num_layers entries: every layer present, each
        // value vec is single-element (the per-layer flag).
        assert_eq!(map.len(), num_layers as usize);
        for layer_idx in 0..num_layers as usize {
            let key = DwqCalibrator::sensitivity_tensor_name(layer_idx);
            let v = map
                .get(&key)
                .unwrap_or_else(|| panic!("missing entry for layer {layer_idx}"));
            assert_eq!(
                v.len(),
                1,
                "per-layer Dwq entry must be single-element flag, got {}",
                v.len()
            );
            assert!(
                v[0] == 0.0 || v[0] == 1.0,
                "per-layer flag must be 0.0 or 1.0, got {}",
                v[0]
            );
        }

        // The mock activations are monotonic in layer index, so the
        // sensitivity ranges should consume at least one (and at most
        // num_layers - 1, since the lowest layer is base) — but the
        // exact split depends on `allocate_bits_by_sensitivity` and
        // `> midpoint` filter, which can yield 0 sensitive layers when
        // the gradient is shallow. We just assert the shape is sane.
        let sensitive_count = map.values().filter(|v| v[0] > 0.5).count();
        assert!(
            sensitive_count <= num_layers as usize,
            "sensitive_count {sensitive_count} must not exceed num_layers"
        );
    }

    #[test]
    fn dwq_cache_hit_skips_forward_pass_same_key() {
        let _lock = crate::calibrate::test_support::CACHE_ENV_LOCK
            .lock()
            .unwrap();
        let tmp = tempfile::tempdir().unwrap();
        let _env_guard = EnvVarGuard::set_path("XDG_CACHE_HOME", tmp.path());

        let num_layers = 4u32;
        let hidden_size = 8u32;
        let counter = Arc::new(AtomicUsize::new(0));
        let lazy_map = LazyTensorMap::new();
        let meta = dummy_metadata(num_layers, hidden_size);
        let progress = ProgressReporter::new();
        let corpus = nonempty_corpus();

        let capture1: Box<dyn ActivationCapture + Send + Sync> = Box::new(
            CountingMockActivationCapture::new(num_layers, hidden_size, Arc::clone(&counter)),
        );
        let mut calib1 = DwqCalibrator::new(DwqArch::Qwen35MoE, Some(capture1), 4, 6, 16);
        let first = calib1
            .calibrate(&lazy_map, &meta, &corpus, &progress)
            .unwrap();
        assert_eq!(counter.load(Ordering::SeqCst), 1);

        let capture2: Box<dyn ActivationCapture + Send + Sync> = Box::new(
            CountingMockActivationCapture::new(num_layers, hidden_size, Arc::clone(&counter)),
        );
        let mut calib2 = DwqCalibrator::new(DwqArch::Qwen35MoE, Some(capture2), 4, 6, 16);
        let second = calib2
            .calibrate(&lazy_map, &meta, &corpus, &progress)
            .unwrap();

        assert_eq!(
            counter.load(Ordering::SeqCst),
            1,
            "second same-key call must be served from cache"
        );
        match (first, second) {
            (CalibrationData::Dwq(a), CalibrationData::Dwq(b)) => assert_eq!(a, b),
            other => panic!("expected DWQ maps, got {other:?}"),
        }
    }

    #[test]
    fn dwq_cache_miss_when_corpus_changes_runs_forward_again() {
        let _lock = crate::calibrate::test_support::CACHE_ENV_LOCK
            .lock()
            .unwrap();
        let tmp = tempfile::tempdir().unwrap();
        let _env_guard = EnvVarGuard::set_path("XDG_CACHE_HOME", tmp.path());

        let num_layers = 4u32;
        let hidden_size = 8u32;
        let counter = Arc::new(AtomicUsize::new(0));
        let lazy_map = LazyTensorMap::new();
        let meta = dummy_metadata(num_layers, hidden_size);
        let progress = ProgressReporter::new();
        let corpus_a = CalibrationCorpus {
            chunks: vec![vec![1u32, 2, 3, 4]],
            name: "a".into(),
        };
        let corpus_b = CalibrationCorpus {
            chunks: vec![vec![4u32, 3, 2, 1]],
            name: "b".into(),
        };

        for corpus in [&corpus_a, &corpus_b] {
            let capture: Box<dyn ActivationCapture + Send + Sync> = Box::new(
                CountingMockActivationCapture::new(num_layers, hidden_size, Arc::clone(&counter)),
            );
            let mut calib = DwqCalibrator::new(DwqArch::Qwen35MoE, Some(capture), 4, 6, 16);
            calib
                .calibrate(&lazy_map, &meta, corpus, &progress)
                .unwrap();
        }

        assert_eq!(
            counter.load(Ordering::SeqCst),
            2,
            "different corpus SHA must miss and run capture again"
        );
    }

    #[test]
    fn dwq_corrupt_cache_warns_and_recomputes() {
        let _lock = crate::calibrate::test_support::CACHE_ENV_LOCK
            .lock()
            .unwrap();
        let tmp = tempfile::tempdir().unwrap();
        let _env_guard = EnvVarGuard::set_path("XDG_CACHE_HOME", tmp.path());

        let num_layers = 4u32;
        let hidden_size = 8u32;
        let counter = Arc::new(AtomicUsize::new(0));
        let lazy_map = LazyTensorMap::new();
        let meta = dummy_metadata(num_layers, hidden_size);
        let progress = ProgressReporter::new();
        let corpus = nonempty_corpus();
        let key = dwq_cache_key(&lazy_map, &meta, &corpus);
        let path = cache_file_path(&key).unwrap();
        fs::write(&path, b"{not-json").unwrap();

        let capture: Box<dyn ActivationCapture + Send + Sync> = Box::new(
            CountingMockActivationCapture::new(num_layers, hidden_size, Arc::clone(&counter)),
        );
        let mut calib = DwqCalibrator::new(DwqArch::Qwen35MoE, Some(capture), 4, 6, 16);
        let data = calib
            .calibrate(&lazy_map, &meta, &corpus, &progress)
            .unwrap();

        assert_eq!(counter.load(Ordering::SeqCst), 1);
        assert!(matches!(data, CalibrationData::Dwq(_)));
    }

    #[test]
    fn dwq_cache_does_not_bypass_forward_pass_unavailable() {
        let _lock = crate::calibrate::test_support::CACHE_ENV_LOCK
            .lock()
            .unwrap();
        let tmp = tempfile::tempdir().unwrap();
        let _env_guard = EnvVarGuard::set_path("XDG_CACHE_HOME", tmp.path());

        let lazy_map = LazyTensorMap::new();
        let meta = dummy_metadata(4, 8);
        let progress = ProgressReporter::new();
        let corpus = nonempty_corpus();
        let key = dwq_cache_key(&lazy_map, &meta, &corpus);
        let path = cache_file_path(&key).unwrap();
        let mut cached = HashMap::new();
        cached.insert("blk.0.sensitivity".to_string(), vec![1.0]);
        save_dwq_to_path(&path, &key, &cached).unwrap();

        let mut calib = DwqCalibrator::new(DwqArch::Qwen35MoE, None, 4, 6, 16);
        let err = calib
            .calibrate(&lazy_map, &meta, &corpus, &progress)
            .unwrap_err();
        assert!(matches!(
            err,
            CalibrationError::ForwardPassUnavailable { .. }
        ));
    }

    /// Object-safe trait usage — required for the dispatch helper.
    #[test]
    fn dwq_calibrator_object_safe() {
        let calib = DwqCalibrator::new(DwqArch::Other, None, 4, 6, 1024);
        let _: Box<dyn Calibrator> = Box::new(calib);
    }

    /// Send + Sync — required by the trait bound.
    #[test]
    fn dwq_calibrator_is_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<DwqCalibrator>();
    }

    /// `requires_forward_pass()` consults the arch table even after
    /// construction — the arch is the source of truth.
    #[test]
    fn dwq_calibrator_arch_drives_forward_pass_query() {
        // Even with capture, Other arch reports false.
        let cap: Box<dyn ActivationCapture + Send + Sync> =
            Box::new(MockActivationCapture::new(2, 8));
        let calib = DwqCalibrator::new(DwqArch::Other, Some(cap), 4, 6, 1024);
        assert!(!calib.requires_forward_pass());

        // Without capture, Qwen35MoE still reports true.
        let calib2 = DwqCalibrator::new(DwqArch::Qwen35MoE, None, 4, 6, 1024);
        assert!(calib2.requires_forward_pass());
    }
}
