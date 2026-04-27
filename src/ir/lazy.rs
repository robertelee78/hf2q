//! Lazy tensor primitive (ADR-014 Decisions 1 + 2).
//!
//! The streaming convert pipeline produces tensors on demand: shape and
//! dtype are known at parse time, but the bytes only materialise at
//! quantize time. This module is the foundation that lifts
//! `read_safetensors → Phase 1.x transforms → quantize → write` from
//! "every byte resident at every stage" to "one tensor in flight, the
//! rest still pages on disk."
//!
//! ## Design contract
//!
//! - [`LazyTensor`] is `FnOnce`-backed: a tensor produces its bytes
//!   exactly once. Re-materialisation is a compile-time error, not a
//!   runtime allocation. This is intentional — the bytes are owned and
//!   dropped by the caller; producing them again would silently double
//!   memory pressure.
//! - [`LazyTensor::shape`] / [`dtype`] / [`name`] are metadata-only; they
//!   never invoke the materialiser. Phase 1.x transforms that only need
//!   metadata (Phase 1.4 prefix-strip is the canonical example) compose
//!   without ever touching the bulk bytes.
//! - [`LazyTensorMap`] uses `BTreeMap` (not `HashMap`) for deterministic
//!   iteration order, required for the byte-identical regression
//!   contract on uncalibrated paths (ADR-014 Decision 17).
//! - All public types are `Send`. Workers in the rayon pool from
//!   ADR-014 P3 (Decision 5) move `LazyTensor`s across threads; the
//!   `Send` bound is verified by [`tests::test_send_bound`].
//!
//! ## Why FnOnce
//!
//! Materialisation produces an owned `Vec<u8>` (or `TensorRef` wrapping
//! one). Re-materialising would re-allocate — that's exactly the
//! double-mmap-tap pathology the lazy IR exists to prevent. Rust's
//! `FnOnce` lets the consume-once invariant be a compile-time error
//! rather than a runtime panic.
//!
//! ## Materialisation sources
//!
//! Three flavours, all carried as the closure body of [`LazyState::Pending`]:
//!
//! | Source | Closure captures |
//! |--------|------------------|
//! | mmap'd safetensors slice | `Arc<Mmap>` + offset + len |
//! | already-resident bytes | `Vec<u8>` (rare; for tests + post-merge spans) |
//! | derived (transform output) | parent `LazyTensor` + `FnOnce` transform |
//!
//! The derived form is what powers `.map()` composition — three Phase
//! 1.x transforms chained on one tensor produce a single
//! `LazyState::Pending` whose closure runs the chain at materialisation
//! time, exactly once.

use std::collections::BTreeMap;
use std::fmt;

use thiserror::Error;

use super::{DType, IrError, TensorMap, TensorRef};

/// Error returned when a lazy materialiser fails.
#[derive(Error, Debug)]
pub enum MaterializeError {
    /// I/O error while reading from a memory-mapped source.
    #[error("I/O error materialising tensor '{name}': {source}")]
    Io {
        name: String,
        #[source]
        source: std::io::Error,
    },

    /// A transform applied via `.map()` returned an error.
    #[error("Transform failed for tensor '{name}': {reason}")]
    Transform { name: String, reason: String },

    /// Bytes produced by the materialiser don't match the declared metadata.
    #[error(
        "Tensor '{name}' materialised with {actual} bytes; declared metadata expects {expected} bytes"
    )]
    SizeMismatch {
        name: String,
        expected: usize,
        actual: usize,
    },

    /// A core IR error surfaced through the materialiser (e.g. dtype
    /// conversion failed inside a derived transform).
    #[error("IR error materialising tensor '{name}': {source}")]
    Ir {
        name: String,
        #[source]
        source: IrError,
    },
}

/// Metadata about a tensor that is queryable without invoking the
/// materialiser. Carries everything callers need to plan transforms,
/// route to the right quantizer, and emit GGUF/safetensors headers.
#[derive(Debug, Clone)]
pub struct LazyMeta {
    /// Fully qualified tensor name (e.g.,
    /// `model.layers.0.self_attn.q_proj.weight`).
    pub name: String,
    /// Shape of the tensor.
    pub shape: Vec<usize>,
    /// Element dtype.
    pub dtype: DType,
    /// Expected byte length on materialisation
    /// (`numel * dtype.element_size()`). Pre-computed so
    /// [`LazyTensor::materialize`] can validate the materialiser without
    /// touching tensor bytes.
    pub byte_len: usize,
}

impl LazyMeta {
    /// Construct from explicit fields, deriving `byte_len` from
    /// `shape × dtype.element_size()`. The constructor is the canonical
    /// way to create metadata; it makes `byte_len` impossible to set
    /// inconsistently with `shape × dtype`.
    pub fn new(name: String, shape: Vec<usize>, dtype: DType) -> Self {
        let numel: usize = shape.iter().product();
        let byte_len = numel * dtype.element_size();
        Self {
            name,
            shape,
            dtype,
            byte_len,
        }
    }
}

/// Internal materialisation state.
///
/// Public API ([`LazyTensor::materialize`], [`LazyTensor::map`], etc.)
/// hides this — callers should not destructure. The enum is exposed at
/// the module-private level so `.map()` composition can construct
/// either variant without going through a constructor that would
/// re-validate already-validated metadata.
enum LazyState {
    /// Bytes already in memory.
    Materialized(Vec<u8>),
    /// Closure that produces bytes on demand. `FnOnce` enforces
    /// consume-once semantics at compile time.
    Pending(Box<dyn FnOnce() -> Result<Vec<u8>, MaterializeError> + Send + 'static>),
}

impl fmt::Debug for LazyState {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            LazyState::Materialized(bytes) => f
                .debug_tuple("Materialized")
                .field(&format_args!("<{} bytes>", bytes.len()))
                .finish(),
            LazyState::Pending(_) => f.debug_tuple("Pending").field(&"<closure>").finish(),
        }
    }
}

/// A tensor whose bytes have not (necessarily) been materialised.
///
/// Carries [`LazyMeta`] (cheap to read; the `shape`/`dtype`/`name`
/// accessors never invoke the materialiser) plus a [`LazyState`] that
/// is either already-resident bytes or a deferred closure.
///
/// Construct via [`LazyTensor::from_bytes`] (already-resident),
/// [`LazyTensor::from_closure`] (pending), or [`LazyTensor::map`]
/// (derived from another LazyTensor).
#[derive(Debug)]
pub struct LazyTensor {
    meta: LazyMeta,
    state: LazyState,
}

impl LazyTensor {
    /// Construct from already-materialised bytes. Used when the input
    /// path is eager (e.g. tests, in-memory transforms applied
    /// pre-stream) and for the `Materialized` form returned by
    /// [`materialize`](Self::materialize) consumers that re-wrap.
    pub fn from_bytes(meta: LazyMeta, bytes: Vec<u8>) -> Self {
        Self {
            meta,
            state: LazyState::Materialized(bytes),
        }
    }

    /// Construct from a deferred closure. The closure produces the raw
    /// bytes for this tensor; `materialize()` calls it exactly once and
    /// validates that the byte count matches `meta.byte_len`.
    pub fn from_closure<F>(meta: LazyMeta, load: F) -> Self
    where
        F: FnOnce() -> Result<Vec<u8>, MaterializeError> + Send + 'static,
    {
        Self {
            meta,
            state: LazyState::Pending(Box::new(load)),
        }
    }

    /// Tensor name. Metadata-only: never invokes the materialiser.
    pub fn name(&self) -> &str {
        &self.meta.name
    }

    /// Tensor shape. Metadata-only: never invokes the materialiser.
    pub fn shape(&self) -> &[usize] {
        &self.meta.shape
    }

    /// Tensor dtype. Metadata-only: never invokes the materialiser.
    pub fn dtype(&self) -> DType {
        self.meta.dtype
    }

    /// Expected byte length on materialisation. Metadata-only.
    pub fn byte_len(&self) -> usize {
        self.meta.byte_len
    }

    /// Borrow the metadata struct. Metadata-only.
    pub fn meta(&self) -> &LazyMeta {
        &self.meta
    }

    /// Materialise the tensor's bytes into a [`TensorRef`].
    ///
    /// Consumes `self`: a `LazyTensor` produces its bytes exactly once.
    /// Re-materialising would silently double memory pressure; the
    /// `FnOnce` move into this method makes that a compile-time error
    /// (verified by [`tests::test_materialize_once_fnonce_compile`]).
    ///
    /// Validates that the produced byte count matches `meta.byte_len`
    /// before constructing the `TensorRef` — a materialiser that returns
    /// the wrong size is a programming error in the producer (e.g. a
    /// stale safetensors header), not something callers should
    /// silently absorb.
    pub fn materialize(self) -> Result<TensorRef, MaterializeError> {
        let LazyTensor { meta, state } = self;
        let bytes = match state {
            LazyState::Materialized(bytes) => bytes,
            LazyState::Pending(load) => load()?,
        };

        if bytes.len() != meta.byte_len {
            return Err(MaterializeError::SizeMismatch {
                name: meta.name,
                expected: meta.byte_len,
                actual: bytes.len(),
            });
        }

        Ok(TensorRef {
            name: meta.name,
            shape: meta.shape,
            dtype: meta.dtype,
            data: bytes,
        })
    }

    /// Compose a shape- and dtype-preserving transform onto this lazy
    /// tensor. The transform runs at materialisation time, exactly once.
    ///
    /// The transform receives a fully-realised [`TensorRef`] and must
    /// return one with the same `shape`, `dtype`, and `byte_len`
    /// (enforced by the size check inside [`materialize`](Self::materialize)).
    /// For shape- or dtype-changing operations (e.g. expert merge, dtype
    /// cast), use [`LazyTensor::map_with_meta`] instead.
    ///
    /// `.map()` composes — three transforms chained produce one
    /// `LazyState::Pending` whose closure runs the entire chain at
    /// materialisation time, never re-allocating intermediate bytes that
    /// are immediately overwritten.
    pub fn map<F>(self, f: F) -> Self
    where
        F: FnOnce(TensorRef) -> Result<TensorRef, MaterializeError> + Send + 'static,
    {
        let new_meta = self.meta.clone();
        let parent = self;
        LazyTensor::from_closure(new_meta, move || {
            let materialized = parent.materialize()?;
            let transformed = f(materialized)?;
            Ok(transformed.data)
        })
    }

    /// Compose a transform that may change shape and/or dtype.
    ///
    /// Caller supplies the post-transform metadata; the transform body
    /// must produce bytes whose length matches `new_meta.byte_len`.
    ///
    /// Used by Phase 1.5 (qwen35moe expert merge — shape changes from
    /// per-expert `[hidden, moe_inter]` to merged
    /// `[N=256, hidden, moe_inter]`) and the eager-pipeline `bf16→f16`
    /// path (dtype changes; `byte_len` is identical only because both
    /// dtypes are 2 bytes — the constructor still computes it from
    /// `shape × dtype.element_size()` so future dtype-size changes can't
    /// silently skip validation).
    pub fn map_with_meta<F>(self, new_meta: LazyMeta, f: F) -> Self
    where
        F: FnOnce(TensorRef) -> Result<TensorRef, MaterializeError> + Send + 'static,
    {
        let parent = self;
        LazyTensor::from_closure(new_meta, move || {
            let materialized = parent.materialize()?;
            let transformed = f(materialized)?;
            Ok(transformed.data)
        })
    }
}

/// Send-bound proof: any closure crossed via `.map()`/`.from_closure()`
/// must be `Send`, and the materialised payload (`Vec<u8>`) is
/// inherently `Send`. Verified at compile time by
/// [`tests::test_send_bound`].
///
// SAFETY: `LazyState::Materialized(Vec<u8>)` is Send. `LazyState::Pending`
// stores a `Box<dyn FnOnce(...) + Send + 'static>` whose `Send` bound is
// declared in the trait object — the auto-trait derivation already covers
// this. We add no unsafe assertion; this comment exists to lock the
// invariant against accidental future drift (e.g. someone making the
// closure non-Send by adding a captured Rc<_>).

/// Map of tensor names to `LazyTensor`s.
///
/// Iteration order is **deterministic** (BTreeMap) — required for the
/// byte-identical regression contract on uncalibrated paths
/// (ADR-014 Decision 17). The previous eager [`TensorMap`] used
/// `HashMap` and called `tensor_names.sort()` at every consumer; the
/// lazy IR enforces sortedness at the data structure level so callers
/// can iterate directly.
#[derive(Debug, Default)]
pub struct LazyTensorMap {
    inner: BTreeMap<String, LazyTensor>,
}

impl LazyTensorMap {
    /// Create an empty map.
    pub fn new() -> Self {
        Self::default()
    }

    /// Insert a lazy tensor. The tensor's `meta.name` is used as the key;
    /// any previous entry with the same name is replaced.
    pub fn insert(&mut self, tensor: LazyTensor) {
        let key = tensor.meta.name.clone();
        self.inner.insert(key, tensor);
    }

    /// Borrow a lazy tensor by name. Metadata-only access; does not
    /// materialise.
    pub fn get(&self, name: &str) -> Option<&LazyTensor> {
        self.inner.get(name)
    }

    /// Remove a lazy tensor by name and return ownership. Used by
    /// transforms that produce a new tensor under a different name
    /// (Phase 1.4 prefix strip), so the old key is consumed and the new
    /// key inserted.
    pub fn remove(&mut self, name: &str) -> Option<LazyTensor> {
        self.inner.remove(name)
    }

    /// Whether a tensor with this name is present. Metadata-only.
    pub fn contains_key(&self, name: &str) -> bool {
        self.inner.contains_key(name)
    }

    /// Number of entries.
    pub fn len(&self) -> usize {
        self.inner.len()
    }

    /// Whether the map is empty.
    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    /// Iterate (name, &LazyTensor) in deterministic order. Metadata-only.
    pub fn iter(&self) -> impl Iterator<Item = (&String, &LazyTensor)> {
        self.inner.iter()
    }

    /// Iterate names in deterministic order. Metadata-only.
    pub fn names(&self) -> impl Iterator<Item = &String> {
        self.inner.keys()
    }

    /// Consume the map, yielding `(name, LazyTensor)` in deterministic
    /// order. The streaming quantize loop (ADR-014 P2) consumes via
    /// this iterator: each tensor is materialised, quantised, written,
    /// and dropped before the next is touched.
    pub fn into_iter(self) -> impl Iterator<Item = (String, LazyTensor)> {
        self.inner.into_iter()
    }

    /// Construct a [`LazyTensorMap`] from an eager [`TensorMap`] —
    /// every tensor's bytes are already resident, so each entry becomes
    /// a [`LazyTensor::from_bytes`] (Materialized variant) at insertion.
    ///
    /// **No streaming benefit on this path.** Callers should use
    /// [`crate::input::safetensors::read_tensors_lazy`] for the streaming
    /// path; `from_eager` is the bridge for cases where a
    /// `TensorMap` already exists (e.g. ADR-014 P2 iter-2's transitional
    /// wiring of `quantize_streaming` into `cmd_convert` paths whose
    /// upstream still produces eager `TensorMap`).
    pub fn from_eager(tensor_map: TensorMap) -> Self {
        let mut out = Self::new();
        for (_, tref) in tensor_map.tensors.into_iter() {
            let meta = LazyMeta::new(tref.name.clone(), tref.shape.clone(), tref.dtype);
            out.insert(LazyTensor::from_bytes(meta, tref.data));
        }
        out
    }

    /// Materialise every tensor and produce an eager [`TensorMap`].
    ///
    /// **Bridge for P0**. ADR-014 Decision 2 specifies this as the
    /// compatibility shim that lets pre-P1 callers (eager Phase 1.x
    /// transforms, the existing GGUF/safetensors backends) continue to
    /// see byte-identical input while the new lazy primitive is the
    /// source of truth. P1 lifts the Phase 1.x transforms onto
    /// `LazyTensorMap`, P2 does the same for the quantize loop, and
    /// `materialize_all()` becomes used only by tests at that point.
    ///
    /// The byte-identical regression test
    /// (`tests/lazy_tensor.rs::test_lazy_safetensors_byte_identical_to_eager`)
    /// uses this method to compare the lazy reader against the
    /// pre-ADR-014 eager reader.
    pub fn materialize_all(self) -> Result<TensorMap, MaterializeError> {
        let mut out = TensorMap::new();
        for (_, lazy) in self.inner.into_iter() {
            let tref = lazy.materialize()?;
            out.insert(tref);
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;

    fn meta(name: &str, shape: Vec<usize>, dtype: DType) -> LazyMeta {
        LazyMeta::new(name.to_string(), shape, dtype)
    }

    fn f32_bytes(values: &[f32]) -> Vec<u8> {
        values.iter().flat_map(|v| v.to_le_bytes()).collect()
    }

    #[test]
    fn test_meta_byte_len_derives_from_shape_and_dtype() {
        let m = meta("t", vec![2, 3], DType::F32);
        assert_eq!(m.byte_len, 2 * 3 * 4);
    }

    #[test]
    fn test_materialize_pending_runs_closure_once() {
        let counter = Arc::new(AtomicUsize::new(0));
        let counter_clone = counter.clone();
        let m = meta("t", vec![3], DType::F32);
        let lazy = LazyTensor::from_closure(m, move || {
            counter_clone.fetch_add(1, Ordering::SeqCst);
            Ok(f32_bytes(&[1.0, 2.0, 3.0]))
        });

        // Closure has not fired yet — proves shape/dtype/name are metadata-only.
        assert_eq!(counter.load(Ordering::SeqCst), 0);
        assert_eq!(lazy.shape(), &[3]);
        assert_eq!(lazy.dtype(), DType::F32);
        assert_eq!(lazy.name(), "t");
        assert_eq!(counter.load(Ordering::SeqCst), 0);

        let tref = lazy.materialize().unwrap();
        assert_eq!(counter.load(Ordering::SeqCst), 1);
        assert_eq!(tref.shape, vec![3]);
        assert_eq!(tref.dtype, DType::F32);
        assert_eq!(tref.data, f32_bytes(&[1.0, 2.0, 3.0]));
    }

    /// Compile-time proof of FnOnce semantics: a `LazyTensor` is moved
    /// into `materialize`, so a second call wouldn't compile. The
    /// commented block below is the negative test — uncommenting it
    /// triggers `error[E0382]: use of moved value`. We can't actually
    /// run a should-not-compile check inside a regular `#[test]`
    /// function; this block is documentation that the contract is
    /// enforced by the type system, not by a runtime panic.
    #[test]
    fn test_materialize_once_fnonce_compile() {
        let m = meta("t", vec![1], DType::F32);
        let lazy = LazyTensor::from_closure(m, || Ok(f32_bytes(&[42.0])));
        let _first = lazy.materialize().unwrap();
        // Uncommenting the next line is a compile error
        // (`use of moved value: 'lazy'`):
        // let _second = lazy.materialize().unwrap();
    }

    #[test]
    fn test_shape_dtype_no_materialise() {
        let counter = Arc::new(AtomicUsize::new(0));
        let counter_clone = counter.clone();
        let m = meta("guard", vec![4, 8], DType::BF16);
        let lazy = LazyTensor::from_closure(m, move || {
            counter_clone.fetch_add(1, Ordering::SeqCst);
            Ok(vec![0u8; 4 * 8 * 2])
        });

        for _ in 0..100 {
            let _ = lazy.shape();
            let _ = lazy.dtype();
            let _ = lazy.name();
            let _ = lazy.byte_len();
            let _ = lazy.meta();
        }
        assert_eq!(
            counter.load(Ordering::SeqCst),
            0,
            "metadata access must never invoke the materialiser"
        );
    }

    #[test]
    fn test_map_compose_idempotent_three_chain() {
        // Eager reference: input * 2 + 1, then negate.
        let input_eager: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
        let eager: Vec<f32> = input_eager
            .iter()
            .map(|x| x * 2.0)
            .map(|x| x + 1.0)
            .map(|x| -x)
            .collect();

        // Lazy chain: same three transforms via .map().
        let m = meta("chain", vec![4], DType::F32);
        let lazy = LazyTensor::from_closure(m, move || Ok(f32_bytes(&input_eager)));

        let lazy = lazy.map(|t| {
            let mut data = t.data;
            for chunk in data.chunks_exact_mut(4) {
                let v = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]) * 2.0;
                chunk.copy_from_slice(&v.to_le_bytes());
            }
            Ok(TensorRef { data, ..t })
        });

        let lazy = lazy.map(|t| {
            let mut data = t.data;
            for chunk in data.chunks_exact_mut(4) {
                let v = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]) + 1.0;
                chunk.copy_from_slice(&v.to_le_bytes());
            }
            Ok(TensorRef { data, ..t })
        });

        let lazy = lazy.map(|t| {
            let mut data = t.data;
            for chunk in data.chunks_exact_mut(4) {
                let v = -f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
                chunk.copy_from_slice(&v.to_le_bytes());
            }
            Ok(TensorRef { data, ..t })
        });

        let realised = lazy.materialize().unwrap();
        let lazy_values: Vec<f32> = realised
            .data
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();

        assert_eq!(lazy_values, eager);
    }

    #[test]
    fn test_map_compose_runs_closure_chain_only_at_materialise() {
        // Each link in the chain increments a different counter.
        let parent_counter = Arc::new(AtomicUsize::new(0));
        let map1_counter = Arc::new(AtomicUsize::new(0));
        let map2_counter = Arc::new(AtomicUsize::new(0));

        let parent_clone = parent_counter.clone();
        let m = meta("watched", vec![1], DType::F32);
        let lazy = LazyTensor::from_closure(m, move || {
            parent_clone.fetch_add(1, Ordering::SeqCst);
            Ok(f32_bytes(&[7.0]))
        });

        let map1_clone = map1_counter.clone();
        let lazy = lazy.map(move |t| {
            map1_clone.fetch_add(1, Ordering::SeqCst);
            Ok(t)
        });

        let map2_clone = map2_counter.clone();
        let lazy = lazy.map(move |t| {
            map2_clone.fetch_add(1, Ordering::SeqCst);
            Ok(t)
        });

        // Pre-materialise: nothing has fired.
        assert_eq!(parent_counter.load(Ordering::SeqCst), 0);
        assert_eq!(map1_counter.load(Ordering::SeqCst), 0);
        assert_eq!(map2_counter.load(Ordering::SeqCst), 0);
        assert_eq!(lazy.shape(), &[1]);
        assert_eq!(parent_counter.load(Ordering::SeqCst), 0);

        let _ = lazy.materialize().unwrap();
        assert_eq!(parent_counter.load(Ordering::SeqCst), 1);
        assert_eq!(map1_counter.load(Ordering::SeqCst), 1);
        assert_eq!(map2_counter.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn test_map_with_meta_changes_shape_and_dtype() {
        // Synthetic shape change: 3-element F32 → 6-element F16 (still 12 bytes).
        let m_in = meta("dtype-cast", vec![3], DType::F32);
        let lazy = LazyTensor::from_closure(m_in, || Ok(f32_bytes(&[1.0, 2.0, 3.0])));

        let m_out = meta("dtype-cast", vec![6], DType::F16);
        let lazy = lazy.map_with_meta(m_out, |t| {
            // Re-interpret 3×f32 → 6×f16 (synthetic, just for the size invariant).
            let mut data = Vec::with_capacity(12);
            for chunk in t.data.chunks_exact(4) {
                let v = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
                let f16 = half::f16::from_f32(v);
                data.extend_from_slice(&f16.to_le_bytes());
                data.extend_from_slice(&f16.to_le_bytes());
            }
            Ok(TensorRef {
                name: t.name,
                shape: vec![6],
                dtype: DType::F16,
                data,
            })
        });

        assert_eq!(lazy.shape(), &[6]);
        assert_eq!(lazy.dtype(), DType::F16);
        let realised = lazy.materialize().unwrap();
        assert_eq!(realised.dtype, DType::F16);
        assert_eq!(realised.data.len(), 12);
    }

    #[test]
    fn test_size_mismatch_is_typed_error() {
        let m = meta("t", vec![4], DType::F32); // expects 16 bytes
        let lazy = LazyTensor::from_closure(m, || Ok(vec![0u8; 8])); // wrong
        let err = lazy.materialize().unwrap_err();
        match err {
            MaterializeError::SizeMismatch {
                expected, actual, ..
            } => {
                assert_eq!(expected, 16);
                assert_eq!(actual, 8);
            }
            _ => panic!("expected SizeMismatch"),
        }
    }

    #[test]
    fn test_materialize_io_error_propagates() {
        let m = meta("io", vec![1], DType::F32);
        let lazy = LazyTensor::from_closure(m, || {
            Err(MaterializeError::Io {
                name: "io".to_string(),
                source: std::io::Error::new(std::io::ErrorKind::NotFound, "synthetic"),
            })
        });
        let err = lazy.materialize().unwrap_err();
        match err {
            MaterializeError::Io { name, .. } => assert_eq!(name, "io"),
            _ => panic!("expected Io"),
        }
    }

    #[test]
    fn test_lazy_tensor_map_btreemap_ordering() {
        let mut map = LazyTensorMap::new();
        // Insert in reverse-alphabetic order; iteration must come back sorted.
        for name in ["zebra", "alpha", "mango", "beta"] {
            let m = meta(name, vec![1], DType::F32);
            map.insert(LazyTensor::from_closure(m, || Ok(f32_bytes(&[0.0]))));
        }
        let names: Vec<&String> = map.names().collect();
        assert_eq!(names, vec!["alpha", "beta", "mango", "zebra"]);
    }

    #[test]
    fn test_lazy_tensor_map_materialize_all_round_trip() {
        let mut map = LazyTensorMap::new();
        let m_a = meta("a", vec![2], DType::F32);
        map.insert(LazyTensor::from_bytes(m_a, f32_bytes(&[1.0, 2.0])));
        let m_b = meta("b", vec![1], DType::F32);
        map.insert(LazyTensor::from_bytes(m_b, f32_bytes(&[42.0])));

        let eager = map.materialize_all().unwrap();
        assert_eq!(eager.len(), 2);
        let a = eager.get("a").unwrap();
        assert_eq!(a.shape, vec![2]);
        assert_eq!(a.data, f32_bytes(&[1.0, 2.0]));
        let b = eager.get("b").unwrap();
        assert_eq!(b.data, f32_bytes(&[42.0]));
    }

    /// Compile-time proof that LazyTensor is Send. If the bound were
    /// removed (e.g. by adding a captured `Rc<_>` to LazyState::Pending),
    /// this function would fail to compile.
    #[test]
    fn test_send_bound() {
        fn assert_send<T: Send>() {}
        assert_send::<LazyTensor>();
        assert_send::<LazyTensorMap>();
        assert_send::<LazyMeta>();
        assert_send::<MaterializeError>();
    }

    #[test]
    fn test_remove_returns_ownership() {
        let mut map = LazyTensorMap::new();
        let m = meta("k", vec![1], DType::F32);
        map.insert(LazyTensor::from_bytes(m, f32_bytes(&[3.14])));
        let owned = map.remove("k").unwrap();
        assert_eq!(owned.name(), "k");
        assert!(map.get("k").is_none());
        let realised = owned.materialize().unwrap();
        assert_eq!(realised.data, f32_bytes(&[3.14]));
    }
}
