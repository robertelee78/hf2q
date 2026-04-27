//! `Calibrator` trait — orthogonal axis of weight-quantization
//! calibrators (ADR-014 P7 Decision 9).
//!
//! ## Why a trait
//!
//! Pre-ADR-014, calibration logic was tangled with the static
//! quantizer hierarchy (`StaticQuantizer`, `MixedBitQuantizer`,
//! `DwqQuantizer`, `ApexQuantizer` in `src/quantize/`). Each
//! quantizer carried its own optional calibration step bundled with
//! its codebook search. ADR-014 P7 splits these axes:
//!
//! - **Calibrator** (this trait): produces per-tensor calibration
//!   data from a model + corpus. Examples: `None` (no calibration),
//!   `Imatrix` (llama.cpp's importance matrix), `Dwq` (Apple/MLX's
//!   distilled weight quantization).
//! - **OutputFormat** (`src/quantize/output_format.rs`, future iter):
//!   the on-disk codebook (Flat / BitPair / KQuant / KQuantAdaptive).
//!
//! The split lets `(Calibrator, OutputFormat)` compose orthogonally
//! — `Imatrix × KQuant` and `Dwq × BitPair` are both reachable, and
//! the CLI (`--quant imatrix-q4_k_m`, `--quant dwq-4-6`) picks the
//! validated cells.
//!
//! ## This iter — minimal Calibrator API
//!
//! P7 iter-1 lands the trait + [`CalibrationData`] enum +
//! [`NoneCalibrator`] impl. The Imatrix and DWQ calibrator impls land
//! in subsequent P7 iters when forward-pass orchestration is wired
//! (uses ADR-013's `RealActivationCapture::run_calibration_prompt`
//! for qwen35/qwen35moe and `gemma4/forward_cpu.rs` for Gemma-4).
//! The `Layout A` path migration of `dwq.rs` / `dwq_activation.rs`
//! / `sensitivity.rs` / `apex.rs` from `src/quantize/` into
//! `src/calibrate/` lands alongside the DwqCalibrator impl.

use std::collections::HashMap;

use sha2::{Digest, Sha256};
use thiserror::Error;

use super::imatrix::Stats as ImatrixStats;

/// Errors from calibration. Wraps the algorithm-specific error types
/// (e.g. [`super::imatrix::ImatrixError`]) under a uniform surface for
/// the calibrator dispatch.
#[derive(Error, Debug)]
pub enum CalibrationError {
    #[error("calibration: forward-pass infrastructure unavailable for arch '{arch}'")]
    ForwardPassUnavailable { arch: String },

    #[error("calibration: corpus is empty")]
    EmptyCorpus,

    #[error("calibration: imatrix algorithm error: {0}")]
    Imatrix(#[from] super::imatrix::ImatrixError),

    #[error("calibration: I/O error: {0}")]
    Io(#[from] std::io::Error),

    #[error("calibration: {message}")]
    Other { message: String },
}

/// Calibration corpus — token sequences fed through the model's
/// forward pass to produce activation captures.
///
/// **Minimal shape this iter**: token sequences only. Full corpus
/// metadata (tokenizer, sampling parameters, batch size, chunk
/// boundaries) lands when the Imatrix/Dwq calibrators wire forward
/// pass — until then, `Calibrator::calibrate(...)` impls treat the
/// corpus as opaque and the actual forward-pass driver lives in the
/// arch-specific path.
#[derive(Debug, Clone)]
pub struct CalibrationCorpus {
    /// Tokenised calibration text. Each inner `Vec<u32>` is one
    /// "chunk" — typically 512 tokens. `chunks.len()` is the number
    /// of forward-pass invocations the calibrator will perform.
    pub chunks: Vec<Vec<u32>>,
    /// Optional human-readable name (e.g. "wikitext-2 test split").
    /// Saved into [`super::imatrix::ImatrixCollector`]'s `.imatrix`
    /// dataset field on emit.
    pub name: String,
}

impl CalibrationCorpus {
    /// Total token count across all chunks. Used by progress
    /// reporting and the `imatrix.chunk_count` GGUF metadata field.
    pub fn total_tokens(&self) -> usize {
        self.chunks.iter().map(|c| c.len()).sum()
    }

    /// Number of chunks (forward-pass invocations).
    pub fn n_chunks(&self) -> usize {
        self.chunks.len()
    }

    /// Whether the corpus has zero usable tokens.
    pub fn is_empty(&self) -> bool {
        self.chunks.is_empty() || self.chunks.iter().all(|c| c.is_empty())
    }
}

/// Return a deterministic metadata-fingerprint hash, not byte-content hash,
/// for cache keying. The tensor tuples are read from `LazyTensorMap` metadata
/// only, in its BTreeMap iteration order, so this never materialises tensor
/// bytes. TODO-future ADR-015 byte-hash upgrade: replace this with a true
/// model-content digest once the streaming byte-hash path lands.
pub fn model_fingerprint(
    model: &crate::ir::lazy::LazyTensorMap,
    meta: &crate::ir::ModelMetadata,
) -> String {
    let mut h = Sha256::new();
    update_hash_str(&mut h, "hf2q-model-fingerprint-v1");
    update_hash_str(&mut h, &meta.architecture);
    h.update(meta.num_layers.to_le_bytes());
    h.update(meta.hidden_size.to_le_bytes());

    for (name, tensor) in model.iter() {
        update_hash_str(&mut h, name);
        h.update((tensor.shape().len() as u64).to_le_bytes());
        for dim in tensor.shape() {
            h.update((*dim as u64).to_le_bytes());
        }
        update_hash_str(&mut h, &format!("{:?}", tensor.dtype()));
    }

    hex::encode(h.finalize())
}

/// SHA-256 over corpus token chunks. Tokens are encoded as little-endian
/// `u32`; chunks are separated by four `0xFF` bytes so adjacent chunks cannot
/// collide with one longer chunk.
pub fn corpus_sha(corpus: &CalibrationCorpus) -> String {
    let mut h = Sha256::new();
    for (idx, chunk) in corpus.chunks.iter().enumerate() {
        if idx > 0 {
            h.update([0xFF_u8; 4]);
        }
        for token in chunk {
            h.update(token.to_le_bytes());
        }
    }
    hex::encode(h.finalize())
}

fn update_hash_str(h: &mut Sha256, value: &str) {
    h.update((value.len() as u64).to_le_bytes());
    h.update(value.as_bytes());
}

/// Calibration data — the per-tensor output of a calibrator. Consumed
/// by the [`crate::quantize::Quantizer`] codebook search at quantize
/// time.
#[derive(Debug, Clone)]
pub enum CalibrationData {
    /// No calibration was performed — the quantizer falls back to
    /// unweighted MSE / round-to-nearest.
    None,

    /// Per-tensor importance vectors (output of
    /// [`super::imatrix::ImatrixCollector::finalise`]). Each `Vec<f32>`
    /// has length `row_size` (dense) or `row_size × n_experts` (MoE,
    /// expert-major).
    Imatrix(HashMap<String, Vec<f32>>),

    /// Per-tensor importance + raw stats (preserves `Stats { values,
    /// counts }` for round-trip through the GGUF format which encodes
    /// per-expert counts exactly, unlike the legacy format's lossy
    /// collapse).
    ImatrixWithStats(HashMap<String, ImatrixStats>),

    /// DWQ sensitivity scores — per-layer scalar importance that
    /// drives the bit-pair allocation in
    /// [`crate::calibrate::dwq::DwqQuantizer`]. Each entry's `Vec<f32>`
    /// is a single value (the layer's sensitivity score) wrapped for
    /// uniform shape with the imatrix variant.
    Dwq(HashMap<String, Vec<f32>>),
}

impl CalibrationData {
    /// Whether this data carries any useful information for the
    /// quantizer (vs. None).
    pub fn is_some(&self) -> bool {
        !matches!(self, CalibrationData::None)
    }

    /// Number of per-tensor entries.
    pub fn len(&self) -> usize {
        match self {
            CalibrationData::None => 0,
            CalibrationData::Imatrix(m) => m.len(),
            CalibrationData::ImatrixWithStats(m) => m.len(),
            CalibrationData::Dwq(m) => m.len(),
        }
    }

    /// Whether the calibration data has zero entries (compatible
    /// with `Self::None` semantics).
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Construct [`CalibrationData::ImatrixWithStats`] from a
    /// llama.cpp-compatible GGUF imatrix file (per the schema landed
    /// in PR #9400 / commit `90083283` / 2025-07-19).
    ///
    /// This is the bridge consumers use to plug a pre-computed
    /// imatrix.gguf into the codec's per-row imatrix-weighted dispatch:
    ///
    /// ```ignore
    /// let calib = CalibrationData::from_imatrix_gguf(&path)?;
    /// let bytes = quantize_row_to_bytes(
    ///     &row, KQuantTarget::Q4K, &calib, "blk.0.attn_q.weight",
    /// )?;
    /// ```
    ///
    /// The returned data uses [`Self::ImatrixWithStats`] (preserves
    /// per-expert counts exactly — the property the GGUF format was
    /// introduced to provide). The codec's
    /// [`crate::quantize::k_quant_codec::quantize_row_to_bytes`]
    /// reads `stats.values.as_slice()` for the imatrix-weighted
    /// codebook search; counts are preserved for round-trip but not
    /// consumed at row-quantize time.
    pub fn from_imatrix_gguf(path: &std::path::Path) -> Result<Self, super::imatrix::ImatrixError> {
        let (collector, _datasets) = super::imatrix::ImatrixCollector::load_imatrix_gguf(path)?;
        Ok(Self::from_imatrix_collector(&collector))
    }

    /// Construct [`CalibrationData::ImatrixWithStats`] from an in-memory
    /// [`super::imatrix::ImatrixCollector`] (e.g. one freshly built via
    /// the `accumulate_*` methods, before saving to disk).
    ///
    /// `stats` is cloned from the collector's internal map so the
    /// collector remains usable post-call.
    pub fn from_imatrix_collector(collector: &super::imatrix::ImatrixCollector) -> Self {
        let stats: HashMap<String, ImatrixStats> = collector
            .stats()
            .iter()
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect();
        Self::ImatrixWithStats(stats)
    }
}

/// Calibrator trait — produces [`CalibrationData`] from a model +
/// calibration corpus.
///
/// **Lifecycle (intended)**:
///
/// 1. Caller builds the calibrator (e.g. `NoneCalibrator::new()`
///    or a future `ImatrixCalibrator::new()`).
/// 2. Caller invokes `calibrate(model, meta, corpus, progress)`.
/// 3. Implementation runs forward pass through the corpus (or
///    no-ops, for `None`), accumulates per-tensor data, and returns
///    [`CalibrationData`].
///
/// **Send + Sync**: calibrators may be invoked from rayon worker
/// threads under a future P3-style parallel calibration loop. The
/// trait bound documents that contract upfront.
///
/// **Forward-pass infrastructure dependency**: implementations that
/// `requires_forward_pass()` use ADR-013's
/// `RealActivationCapture::run_calibration_prompt` for qwen35
/// /qwen35moe and `gemma4/forward_cpu.rs` for Gemma-4. The arch
/// dispatch happens inside the trait impl; the orchestration shell
/// (this trait) is arch-agnostic.
pub trait Calibrator: Send + Sync {
    /// Human-readable calibrator name. Used by logs, the `--quant`
    /// CLI variant resolver (P8), and the imatrix sidecar's
    /// dataset metadata field.
    fn name(&self) -> &'static str;

    /// Whether this calibrator needs a forward pass through the
    /// model. `false` → no-op calibrator (`None`); `true` → reads
    /// activations during forward (`Imatrix`, `Dwq`).
    fn requires_forward_pass(&self) -> bool;

    /// Run the calibration. The exact behaviour depends on the
    /// implementor.
    fn calibrate(
        &mut self,
        model: &crate::ir::lazy::LazyTensorMap,
        meta: &crate::ir::ModelMetadata,
        corpus: &CalibrationCorpus,
        progress: &crate::progress::ProgressReporter,
    ) -> Result<CalibrationData, CalibrationError>;
}

/// `NoneCalibrator` — the no-op calibrator. Returns
/// [`CalibrationData::None`] without touching the model or corpus.
/// Used for uncalibrated quantization paths (`--quant q4_k_m` without
/// imatrix, all the static-quant variants).
#[derive(Debug, Default, Clone, Copy)]
pub struct NoneCalibrator;

impl NoneCalibrator {
    /// Construct.
    pub fn new() -> Self {
        Self
    }
}

impl Calibrator for NoneCalibrator {
    fn name(&self) -> &'static str {
        "none"
    }

    fn requires_forward_pass(&self) -> bool {
        false
    }

    fn calibrate(
        &mut self,
        _model: &crate::ir::lazy::LazyTensorMap,
        _meta: &crate::ir::ModelMetadata,
        _corpus: &CalibrationCorpus,
        _progress: &crate::progress::ProgressReporter,
    ) -> Result<CalibrationData, CalibrationError> {
        // Truly no-op. Doesn't even validate the corpus is non-empty
        // — `None` calibration is the path explicitly chosen when
        // calibration would be unwanted overhead (e.g. running a
        // smoke convert with q4 on a small fixture).
        Ok(CalibrationData::None)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;

    use crate::ir::lazy::{LazyMeta, LazyTensor, LazyTensorMap};
    use crate::ir::DType;

    fn dummy_metadata() -> crate::ir::ModelMetadata {
        crate::ir::ModelMetadata {
            architecture: "Test".into(),
            model_type: "test".into(),
            param_count: 0,
            hidden_size: 4,
            num_layers: 1,
            layer_types: vec![],
            num_attention_heads: 1,
            num_kv_heads: Some(1),
            vocab_size: 16,
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

    fn lazy_tensor(name: &str, shape: Vec<usize>, dtype: DType) -> LazyTensor {
        let meta = LazyMeta::new(name.to_string(), shape, dtype);
        LazyTensor::from_bytes(meta.clone(), vec![0_u8; meta.byte_len])
    }

    fn map_with_tensors(tensors: Vec<LazyTensor>) -> LazyTensorMap {
        let mut map = LazyTensorMap::new();
        for tensor in tensors {
            map.insert(tensor);
        }
        map
    }

    #[test]
    fn none_calibrator_is_no_op() {
        let mut calibrator = NoneCalibrator::new();
        assert_eq!(calibrator.name(), "none");
        assert!(!calibrator.requires_forward_pass());

        let lazy_map = LazyTensorMap::new();
        let metadata = dummy_metadata();
        let corpus = CalibrationCorpus {
            chunks: vec![vec![1, 2, 3, 4]],
            name: "test".into(),
        };
        let progress = crate::progress::ProgressReporter::new();

        let data = calibrator
            .calibrate(&lazy_map, &metadata, &corpus, &progress)
            .unwrap();
        match data {
            CalibrationData::None => {}
            other => panic!("expected CalibrationData::None, got {other:?}"),
        }
    }

    /// `NoneCalibrator` ignores the corpus entirely — even an empty
    /// corpus is accepted (no `EmptyCorpus` error). This matches the
    /// "truly no-op" contract.
    #[test]
    fn none_calibrator_accepts_empty_corpus() {
        let mut calibrator = NoneCalibrator::new();
        let lazy_map = LazyTensorMap::new();
        let metadata = dummy_metadata();
        let corpus = CalibrationCorpus {
            chunks: vec![],
            name: "empty".into(),
        };
        let progress = crate::progress::ProgressReporter::new();
        assert!(calibrator
            .calibrate(&lazy_map, &metadata, &corpus, &progress)
            .is_ok());
    }

    #[test]
    fn calibration_data_is_some_predicate() {
        assert!(!CalibrationData::None.is_some());
        assert!(CalibrationData::Imatrix(HashMap::new()).is_some());
        assert!(CalibrationData::ImatrixWithStats(HashMap::new()).is_some());
        assert!(CalibrationData::Dwq(HashMap::new()).is_some());
    }

    #[test]
    fn calibration_data_len_and_empty() {
        assert_eq!(CalibrationData::None.len(), 0);
        assert!(CalibrationData::None.is_empty());

        let mut m = HashMap::new();
        m.insert("a".to_string(), vec![1.0_f32]);
        m.insert("b".to_string(), vec![2.0_f32]);
        let data = CalibrationData::Imatrix(m);
        assert_eq!(data.len(), 2);
        assert!(!data.is_empty());
    }

    #[test]
    fn calibration_corpus_introspection() {
        let c = CalibrationCorpus {
            chunks: vec![vec![1, 2, 3], vec![4, 5]],
            name: "test".into(),
        };
        assert_eq!(c.total_tokens(), 5);
        assert_eq!(c.n_chunks(), 2);
        assert!(!c.is_empty());

        let empty = CalibrationCorpus {
            chunks: vec![],
            name: "empty".into(),
        };
        assert!(empty.is_empty());
        assert_eq!(empty.total_tokens(), 0);
        assert_eq!(empty.n_chunks(), 0);

        // Non-empty chunks vec but all chunks empty — also is_empty.
        let all_empty_chunks = CalibrationCorpus {
            chunks: vec![vec![], vec![]],
            name: "x".into(),
        };
        assert!(all_empty_chunks.is_empty());
        assert_eq!(all_empty_chunks.total_tokens(), 0);
        assert_eq!(all_empty_chunks.n_chunks(), 2);
    }

    #[test]
    fn corpus_sha_is_deterministic_for_same_chunks() {
        let a = CalibrationCorpus {
            chunks: vec![vec![1, 2, 3], vec![4]],
            name: "a".into(),
        };
        let b = CalibrationCorpus {
            chunks: vec![vec![1, 2, 3], vec![4]],
            name: "different-name-ignored".into(),
        };
        assert_eq!(corpus_sha(&a), corpus_sha(&b));
        assert_eq!(corpus_sha(&a).len(), 64);
        assert!(corpus_sha(&a).chars().all(|c| c.is_ascii_hexdigit()));
    }

    #[test]
    fn corpus_sha_separates_chunk_boundaries() {
        let chunked = CalibrationCorpus {
            chunks: vec![vec![1, 2], vec![3, 4]],
            name: "chunked".into(),
        };
        let flat = CalibrationCorpus {
            chunks: vec![vec![1, 2, 3, 4]],
            name: "flat".into(),
        };
        assert_ne!(corpus_sha(&chunked), corpus_sha(&flat));
    }

    #[test]
    fn corpus_sha_uses_little_endian_u32_tokens() {
        let corpus = CalibrationCorpus {
            chunks: vec![vec![0x0102_0304]],
            name: "endian".into(),
        };
        let mut h = Sha256::new();
        h.update([0x04, 0x03, 0x02, 0x01]);
        assert_eq!(corpus_sha(&corpus), hex::encode(h.finalize()));
    }

    #[test]
    fn model_fingerprint_is_deterministic_in_btree_order() {
        let meta = dummy_metadata();
        let map_a = map_with_tensors(vec![
            lazy_tensor("b.weight", vec![2, 2], DType::F16),
            lazy_tensor("a.weight", vec![4], DType::F32),
        ]);
        let map_b = map_with_tensors(vec![
            lazy_tensor("a.weight", vec![4], DType::F32),
            lazy_tensor("b.weight", vec![2, 2], DType::F16),
        ]);

        assert_eq!(
            model_fingerprint(&map_a, &meta),
            model_fingerprint(&map_b, &meta)
        );
    }

    #[test]
    fn model_fingerprint_changes_with_metadata_shape_or_dtype() {
        let meta = dummy_metadata();
        let map = map_with_tensors(vec![lazy_tensor("w", vec![2, 2], DType::F16)]);
        let shape_changed = map_with_tensors(vec![lazy_tensor("w", vec![4, 1], DType::F16)]);
        let dtype_changed = map_with_tensors(vec![lazy_tensor("w", vec![2, 2], DType::BF16)]);
        let mut meta_changed = dummy_metadata();
        meta_changed.hidden_size += 1;

        let base = model_fingerprint(&map, &meta);
        assert_ne!(base, model_fingerprint(&shape_changed, &meta));
        assert_ne!(base, model_fingerprint(&dtype_changed, &meta));
        assert_ne!(base, model_fingerprint(&map, &meta_changed));
    }

    #[test]
    fn model_fingerprint_does_not_materialise_lazy_tensors() {
        let counter = Arc::new(AtomicUsize::new(0));
        let counter_clone = Arc::clone(&counter);
        let meta = LazyMeta::new("pending.weight".to_string(), vec![2], DType::F32);
        let lazy = LazyTensor::from_closure(meta, move || {
            counter_clone.fetch_add(1, Ordering::SeqCst);
            Ok(vec![0_u8; 8])
        });
        let map = map_with_tensors(vec![lazy]);

        let _ = model_fingerprint(&map, &dummy_metadata());
        assert_eq!(counter.load(Ordering::SeqCst), 0);
    }

    /// Trait is object-safe (can be used as `&dyn Calibrator` /
    /// `Box<dyn Calibrator>`) — required for runtime dispatch from
    /// the CLI variant resolver.
    #[test]
    fn calibrator_is_object_safe() {
        let _: Box<dyn Calibrator> = Box::new(NoneCalibrator::new());
    }

    /// Trait + impls are Send + Sync — required for rayon
    /// parallelisation and shared `Arc<dyn Calibrator>` patterns.
    #[test]
    fn calibrator_is_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<NoneCalibrator>();
        assert_send_sync::<Box<dyn Calibrator>>();
    }

    // ─────────── from_imatrix_collector / from_imatrix_gguf bridge ───────────

    /// `from_imatrix_collector` clones every entry's `Stats` exactly,
    /// produces `CalibrationData::ImatrixWithStats`.
    #[test]
    fn from_imatrix_collector_preserves_stats() {
        use crate::calibrate::imatrix::ImatrixCollector;

        let mut col = ImatrixCollector::new();
        col.accumulate_dense("blk.0.attn_q.weight", &[1.0, 2.0, 3.0, 4.0], 4, 1)
            .unwrap();
        col.record_chunk();

        let calib = CalibrationData::from_imatrix_collector(&col);
        assert!(calib.is_some());
        match &calib {
            CalibrationData::ImatrixWithStats(map) => {
                let stats = map.get("blk.0.attn_q.weight").unwrap();
                let orig = col.stats().get("blk.0.attn_q.weight").unwrap();
                assert_eq!(stats.values, orig.values);
                assert_eq!(stats.counts, orig.counts);
            }
            other => panic!("expected ImatrixWithStats, got {other:?}"),
        }
    }

    /// **Full bridge round trip**: build an ImatrixCollector → save
    /// to GGUF → `CalibrationData::from_imatrix_gguf` → verify the
    /// resulting `ImatrixWithStats` matches the original collector
    /// stats byte-for-byte.
    #[test]
    fn from_imatrix_gguf_save_load_round_trip() {
        use crate::calibrate::imatrix::ImatrixCollector;

        let mut col = ImatrixCollector::new();
        col.accumulate_dense("blk.0.attn_q.weight", &[1.0, 2.0, 3.0, 4.0], 4, 1)
            .unwrap();
        col.accumulate_dense(
            "blk.1.ffn_down.weight",
            &[0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
            6,
            1,
        )
        .unwrap();
        col.record_chunk();

        let dir = std::env::temp_dir().join("hf2q-calibration-bridge-test");
        let _ = std::fs::create_dir_all(&dir);
        let path = dir.join("test.imatrix.gguf");
        let _ = std::fs::remove_file(&path);

        col.save_imatrix_gguf(&path, 512, &["wikitext-2.txt"])
            .unwrap();

        let calib = CalibrationData::from_imatrix_gguf(&path).unwrap();
        let _ = std::fs::remove_file(&path);

        match calib {
            CalibrationData::ImatrixWithStats(map) => {
                assert_eq!(map.len(), 2);
                let q_stats = map.get("blk.0.attn_q.weight").unwrap();
                let q_orig = col.stats().get("blk.0.attn_q.weight").unwrap();
                assert_eq!(q_stats.values, q_orig.values);
                assert_eq!(q_stats.counts, q_orig.counts);

                let ffn_stats = map.get("blk.1.ffn_down.weight").unwrap();
                let ffn_orig = col.stats().get("blk.1.ffn_down.weight").unwrap();
                assert_eq!(ffn_stats.values, ffn_orig.values);
                assert_eq!(ffn_stats.counts, ffn_orig.counts);
            }
            other => panic!("expected ImatrixWithStats, got {other:?}"),
        }
    }

    /// `from_imatrix_gguf` on a missing file surfaces the underlying
    /// I/O error wrapped in `ImatrixError::Io`.
    #[test]
    fn from_imatrix_gguf_missing_file_errors() {
        let path = std::env::temp_dir().join("hf2q-bridge-nonexistent.imatrix.gguf");
        let _ = std::fs::remove_file(&path);
        let err = CalibrationData::from_imatrix_gguf(&path).unwrap_err();
        match err {
            crate::calibrate::imatrix::ImatrixError::Io(_) => {}
            other => panic!("expected Io error, got {other:?}"),
        }
    }

    /// `CalibrationData` round-tripped through the GGUF bridge can be
    /// passed directly to `quantize_row_to_bytes` and produce the
    /// imatrix-weighted bytes (not the `_ref` bytes). Verifies the
    /// codec's `lookup_imatrix_weights` reads `stats.values` correctly.
    #[test]
    fn from_imatrix_gguf_bridges_to_codec() {
        use crate::calibrate::imatrix::ImatrixCollector;
        use crate::quantize::k_quant_codec::{quantize_row_to_bytes, KQuantTarget};

        const QK_K: usize = 256;

        // Build calibration with biased weights (100× emphasis on first 64 cols).
        let mut col = ImatrixCollector::new();
        let mut activations = vec![0.5_f32; QK_K];
        // Repeated values ensure the imatrix weights = activations²
        // gives a clear bias signal.
        for v in activations.iter_mut().take(64) {
            *v = 10.0;
        }
        col.accumulate_dense("test.weight", &activations, 1, QK_K)
            .unwrap();
        col.record_chunk();

        let dir = std::env::temp_dir().join("hf2q-bridge-codec-test");
        let _ = std::fs::create_dir_all(&dir);
        let path = dir.join("test.imatrix.gguf");
        let _ = std::fs::remove_file(&path);
        col.save_imatrix_gguf(&path, 512, &["test.txt"]).unwrap();

        let calib = CalibrationData::from_imatrix_gguf(&path).unwrap();
        let _ = std::fs::remove_file(&path);

        // Quantize the same input row through the codec via the imatrix path.
        let row: Vec<f32> = (0..QK_K).map(|i| (i as f32 - 128.0) / 128.0).collect();
        let bytes_imatrix =
            quantize_row_to_bytes(&row, KQuantTarget::Q4K, &calib, "test.weight").unwrap();
        let bytes_ref = quantize_row_to_bytes(
            &row,
            KQuantTarget::Q4K,
            &CalibrationData::None,
            "test.weight",
        )
        .unwrap();

        assert_eq!(bytes_imatrix.len(), bytes_ref.len(), "same on-disk size");
        // The imatrix-weighted codebook search differs from _ref on
        // biased input → bytes must differ.
        assert_ne!(
            bytes_imatrix, bytes_ref,
            "imatrix path produced byte-identical output to _ref via the bridge"
        );
    }
}
