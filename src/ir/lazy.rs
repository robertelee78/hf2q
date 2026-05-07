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
    /// Bytes already in memory, behind a refcounted pointer.  ADR-014 P13
    /// step 1 (iter-76): added so `quantize_via_streaming_borrowed` and
    /// other transitional wedges can refcount-share bytes from the
    /// caller's `tensor_map` rather than deep-cloning per-tensor.  The
    /// `Arc` is consumed at materialise time — we unwrap or clone the
    /// inner `Vec<u8>` depending on whether refcount==1.  Equivalent to
    /// `Materialized` from the streaming consumer's perspective, but
    /// the upstream caller can hand multiple `LazyTensor`s a shared
    /// `Arc<Vec<u8>>` without paying N×bytes.
    MaterializedShared(std::sync::Arc<Vec<u8>>),
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
            LazyState::MaterializedShared(bytes) => f
                .debug_tuple("MaterializedShared")
                .field(&format_args!("<{} bytes, refcount {}>", bytes.len(), std::sync::Arc::strong_count(bytes)))
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

    /// ADR-014 P13 step 1 (iter-76) — construct from refcount-shared
    /// bytes.  Used by transitional wedges
    /// ([`crate::quantize::quantize_via_streaming_borrowed`] et al.)
    /// that want to feed the streaming pipeline without paying a
    /// per-tensor deep clone.
    ///
    /// The `Arc<Vec<u8>>` allows the same bytes to be shared by
    /// multiple LazyTensors (or stay live in the caller's `TensorRef`
    /// while the LazyTensor is in flight).  At materialise time, if
    /// refcount==1 the inner `Vec<u8>` is unwrapped (zero copy); if >1
    /// the inner is cloned (one copy).
    ///
    /// Memory profile vs [`Self::from_bytes`]: same when refcount==1
    /// (both move the Vec through); strictly better when refcount>1
    /// because the Arc-shared variant amortises the bytes across all
    /// holders, while `from_bytes` requires the caller to clone
    /// upfront.
    ///
    /// The full P13 win lands when [`crate::ir::TensorRef::data`]
    /// itself becomes `Arc<[u8]>` so the upstream tensor_map can
    /// hand `Arc::clone(&t.data)` to the wedge.  This constructor is
    /// the API foothold for that future migration.
    pub fn from_arc_bytes(meta: LazyMeta, bytes: std::sync::Arc<Vec<u8>>) -> Self {
        Self {
            meta,
            state: LazyState::MaterializedShared(bytes),
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
            LazyState::MaterializedShared(arc) => {
                // refcount==1 → unwrap zero-copy; >1 → clone the inner
                // Vec.  Cf. `Arc::try_unwrap` doc; we use the simpler
                // `Arc::unwrap_or_clone` (stable since 1.76) for the
                // exact "unwrap if sole owner else clone" semantic.
                std::sync::Arc::unwrap_or_clone(arc)
            }
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
            data: std::sync::Arc::new(bytes),
        })
    }

    /// Materialise a by-reference copy of an already-resident tensor.
    ///
    /// This is intentionally narrower than [`Self::materialize`]: it only
    /// supports already-resident entries ([`LazyState::Materialized`] or
    /// [`LazyState::MaterializedShared`]) and returns a typed error for
    /// pending `FnOnce` entries. The streaming contract remains intact for
    /// mmap-backed tensors; this exists for bridge paths whose caller
    /// already chose to keep bytes resident and must pass a
    /// `&LazyTensorMap` through an existing trait surface.
    ///
    /// ADR-020 iter-6: the `MaterializedShared` arm hands the same
    /// `Arc<Vec<u8>>` over to the returned `TensorRef` (pointer-bump
    /// share), instead of deep-cloning the inner `Vec`. The original
    /// `LazyTensor` keeps its Arc clone — by-reference semantic
    /// preserved. On 27B-class models this saves ~52 GB peak per
    /// `clone_tensor_map_to_lazy → load_lazy_f32` pass. The
    /// `Materialized` arm still must clone the inner `Vec` because that
    /// variant doesn't share — it's the one-owner form used by tests
    /// and post-merge spans (cf. lazy.rs module-level docs).
    pub fn materialize_cloned(&self) -> Result<TensorRef, MaterializeError> {
        let data: std::sync::Arc<Vec<u8>> = match &self.state {
            LazyState::Materialized(bytes) => std::sync::Arc::new(bytes.clone()),
            LazyState::MaterializedShared(arc) => std::sync::Arc::clone(arc),
            LazyState::Pending(_) => {
                return Err(MaterializeError::Transform {
                    name: self.meta.name.clone(),
                    reason: "borrowed materialization is only available for already-resident tensors"
                        .to_string(),
                });
            }
        };

        if data.len() != self.meta.byte_len {
            return Err(MaterializeError::SizeMismatch {
                name: self.meta.name.clone(),
                expected: self.meta.byte_len,
                actual: data.len(),
            });
        }

        Ok(TensorRef {
            name: self.meta.name.clone(),
            shape: self.meta.shape.clone(),
            dtype: self.meta.dtype,
            data,
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
            Ok(std::sync::Arc::unwrap_or_clone(transformed.data))
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
            Ok(std::sync::Arc::unwrap_or_clone(transformed.data))
        })
    }
}

// Send-bound proof: any closure crossed via `.map()`/`.from_closure()`
// must be `Send`, and the materialised payload (`Vec<u8>`) is
// inherently `Send`. Verified at compile time by
// [`tests::test_send_bound`].
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

    /// ADR-014 P7 iter-55 — convert all BF16 tensors to F16 in-place.
    /// Mirrors `TensorMap::convert_bf16_to_f16` but operates lazily:
    /// each BF16 tensor is replaced with a `LazyTensor` that
    /// materialises → casts → drops the BF16 byte buffer per tensor.
    ///
    /// The cast itself is deferred to materialisation time — calling
    /// this method only rebuilds the closure stack and updates the
    /// meta dtype.  Memory profile during the call is metadata-only;
    /// the per-tensor BF16→F16 cost lands at materialise.
    ///
    /// Required by iter-3 wholesale-skip plan (iter-53 survey) for
    /// `main.rs:1188`'s `tensor_map.convert_bf16_to_f16()` call site:
    ///   tensor_map.convert_bf16_to_f16() → lazy_map.convert_bf16_to_f16()
    ///
    /// Returns the count of converted tensors (matches eager variant's
    /// signature exactly).
    ///
    /// **Byte-equality contract** to `TensorMap::convert_bf16_to_f16`:
    /// after both paths complete, `materialize_all()` produces a
    /// TensorMap with the same set of (name, F16 bytes) pairs.
    /// Verified by `convert_bf16_to_f16_matches_eager`.
    pub fn convert_bf16_to_f16(&mut self) -> Result<usize, MaterializeError> {
        let bf16_names: Vec<String> = self
            .inner
            .iter()
            .filter(|(_, t)| t.meta().dtype == DType::BF16)
            .map(|(name, _)| name.clone())
            .collect();
        let count = bf16_names.len();

        for name in bf16_names {
            if let Some(lazy) = self.inner.remove(&name) {
                let meta = lazy.meta();
                let new_meta = LazyMeta::new(meta.name.clone(), meta.shape.clone(), DType::F16);
                let new_lazy = lazy.map_with_meta(new_meta, |tensor| {
                    tensor.to_f16().map_err(|e| MaterializeError::Transform {
                        name: tensor.name.clone(),
                        reason: format!("bf16→f16: {e}"),
                    })
                });
                self.inner.insert(name, new_lazy);
            }
        }
        Ok(count)
    }

    /// ADR-014 P7 iter-54 — total bytes across all tensors, computed
    /// metadata-only.  Mirrors `TensorMap::total_size_bytes` but
    /// without requiring materialisation: each `LazyMeta::byte_len`
    /// is pre-computed at construction (`numel × dtype.element_size()`)
    /// per `LazyMeta::new` invariant.
    ///
    /// Required by the iter-3 wholesale-skip plan (iter-53 survey)
    /// for `main.rs:1166`'s telemetry call site:
    /// `tensor_map.total_size_bytes()` → `lazy_map.total_size_bytes()`.
    ///
    /// Byte-equality to `TensorMap::total_size_bytes` after a round-trip
    /// through `materialize_all()` is verified by
    /// `total_size_bytes_matches_post_materialize`.
    pub fn total_size_bytes(&self) -> usize {
        self.inner.values().map(|t| t.meta().byte_len).sum()
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
            out.insert(LazyTensor::from_arc_bytes(meta, tref.data));
        }
        out
    }

    /// Construct a metadata-only [`LazyTensorMap`] view of an eager
    /// [`TensorMap`] by reference — every entry becomes a
    /// [`LazyTensor::from_closure`] backed by a closure that *would*
    /// re-clone the bytes if asked, but only the metadata (shape,
    /// dtype, name) is needed for the calibrator-fingerprint
    /// downstream. The source `tensor_map` retains ownership and no
    /// bytes are allocated unless `materialize()` is called.
    ///
    /// **When to use over `from_eager`**: this method is the bridge
    /// when the caller wants to feed [`crate::calibrate::calibrator::Calibrator::calibrate`]
    /// a `LazyTensorMap` view (for the `model_fingerprint` cache key)
    /// without consuming or duplicating the source `tensor_map`. The
    /// fingerprint walks `iter()` (metadata-only) so the closure body
    /// is never invoked in the production calibrate path. ADR-014 P2
    /// iter-2 §S1: used by `cmd_convert` to drive the calibrator
    /// without consuming `tensor_map`.
    ///
    /// **Memory cost**: zero — closures capture nothing because the
    /// production calibrator never materialises this view. If a future
    /// caller does materialise (e.g. a calibrator that itself reads
    /// weight bytes), the closure surfaces a typed
    /// [`MaterializeError::Transform`] rather than silently allocating
    /// — that's a contract change, not a runtime path.
    pub fn from_eager_borrowed(tensor_map: &TensorMap) -> Self {
        let mut out = Self::new();
        for (_, tref) in tensor_map.tensors.iter() {
            let meta = LazyMeta::new(tref.name.clone(), tref.shape.clone(), tref.dtype);
            let name_for_err = tref.name.clone();
            out.insert(LazyTensor::from_closure(meta, move || {
                Err(MaterializeError::Transform {
                    name: name_for_err.clone(),
                    reason: "from_eager_borrowed view is metadata-only; \
                             source tensor_map owns the bytes (ADR-014 P2 iter-2 §S1)"
                        .to_string(),
                })
            }));
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

    /// ADR-020 P13 step 4 falsifier — proves `Arc::clone(&tensor.data)`
    /// + `LazyTensor::from_arc_bytes` is a pointer-bump share, not a
    /// byte clone. Pre-refactor `clone_tensor_map_to_lazy` deep-cloned
    /// each tensor's `Vec<u8>` via `tensor.data.clone()` — the 52 GB
    /// clone for Qwen3.6-27B that dominated the 199 GB DWQ OOM peak.
    ///
    /// The assertion: `Arc::strong_count(&tensor.data) == 2` after the
    /// share. One ref in `tensor.data`, one in the LazyTensor's
    /// `MaterializedShared(arc)` state. Same byte allocation.
    #[test]
    fn arc_clone_lazy_view_shares_bytes_with_source_no_byte_copy() {
        // Build a TensorRef with non-trivial bytes so a deep clone would
        // be observable (different allocation address).
        let bytes = f32_bytes(&[1.0, 2.0, 3.0, 4.0]); // 16 bytes
        let source_data_arc: Arc<Vec<u8>> = Arc::new(bytes);
        let source_addr = Arc::as_ptr(&source_data_arc);
        let tensor = TensorRef {
            name: "shared".to_string(),
            shape: vec![4],
            dtype: DType::F32,
            data: Arc::clone(&source_data_arc),
        };
        // refcount: source_data_arc + tensor.data
        assert_eq!(
            Arc::strong_count(&source_data_arc),
            2,
            "before LazyTensor share: source + tensor"
        );

        // Mirror what `clone_tensor_map_to_lazy` does at main.rs:482-500.
        let lazy = LazyTensor::from_arc_bytes(
            LazyMeta::new(
                tensor.name.clone(),
                tensor.shape.clone(),
                tensor.dtype,
            ),
            Arc::clone(&tensor.data),
        );

        // refcount: source_data_arc + tensor.data + lazy's MaterializedShared
        assert_eq!(
            Arc::strong_count(&source_data_arc),
            3,
            "after LazyTensor::from_arc_bytes: source + tensor + lazy_view all share \
             the SAME Arc — no byte copy. Pre-refactor this would have been a deep \
             clone of the Vec<u8>."
        );

        // Verify the actual allocation hasn't moved (proves no byte copy).
        let after_addr = Arc::as_ptr(&source_data_arc);
        assert_eq!(
            source_addr, after_addr,
            "Vec<u8> allocation address must not change — Arc share is pointer-bump only"
        );

        // Drop lazy_view → refcount goes back to 2. This is the post-
        // calibrate ordering at main.rs:1525 (lazy_view drops at end of
        // match arm) → main.rs:1549 (calibrator drops) → Phase 3 reads
        // tensor.data with refcount=1 (sole owner; could try_unwrap
        // for zero-copy if needed).
        drop(lazy);
        assert_eq!(
            Arc::strong_count(&source_data_arc),
            2,
            "after lazy_view drops: refcount returns to source + tensor"
        );
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
        assert_eq!(*tref.data, f32_bytes(&[1.0, 2.0, 3.0]));
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
            let mut data = std::sync::Arc::unwrap_or_clone(t.data);
            for chunk in data.chunks_exact_mut(4) {
                let v = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]) * 2.0;
                chunk.copy_from_slice(&v.to_le_bytes());
            }
            Ok(TensorRef { data: std::sync::Arc::new(data), ..t })
        });

        let lazy = lazy.map(|t| {
            let mut data = std::sync::Arc::unwrap_or_clone(t.data);
            for chunk in data.chunks_exact_mut(4) {
                let v = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]) + 1.0;
                chunk.copy_from_slice(&v.to_le_bytes());
            }
            Ok(TensorRef { data: std::sync::Arc::new(data), ..t })
        });

        let lazy = lazy.map(|t| {
            let mut data = std::sync::Arc::unwrap_or_clone(t.data);
            for chunk in data.chunks_exact_mut(4) {
                let v = -f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
                chunk.copy_from_slice(&v.to_le_bytes());
            }
            Ok(TensorRef { data: std::sync::Arc::new(data), ..t })
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
                data: std::sync::Arc::new(data),
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

    /// ADR-014 P7 iter-54 — `LazyTensorMap::total_size_bytes` matches
    /// `TensorMap::total_size_bytes` after a round-trip through
    /// `materialize_all()`.  Mid-step in the iter-3 wholesale-skip
    /// plan: this is the metadata-only telemetry helper that lets
    /// `main.rs:1166` swap from eager to lazy without changing any
    /// measured values.
    ///
    /// Multi-tensor + multi-dtype fixture: F32, F16, BF16, U8 — covers
    /// every dtype's `element_size()` cell.
    #[test]
    fn test_total_size_bytes_matches_post_materialize() {
        let mut lazy_map = LazyTensorMap::new();
        // F32: 4 bytes × 2 elems = 8
        lazy_map.insert(LazyTensor::from_bytes(
            meta("a", vec![2], DType::F32),
            vec![0u8; 8],
        ));
        // F16: 2 bytes × 6 elems = 12
        lazy_map.insert(LazyTensor::from_bytes(
            meta("b", vec![3, 2], DType::F16),
            vec![0u8; 12],
        ));
        // BF16: 2 bytes × 4 elems = 8
        lazy_map.insert(LazyTensor::from_bytes(
            meta("c", vec![4], DType::BF16),
            vec![0u8; 8],
        ));
        // U8: 1 byte × 5 elems = 5
        lazy_map.insert(LazyTensor::from_bytes(
            meta("d", vec![5], DType::U8),
            vec![0u8; 5],
        ));

        let lazy_total = lazy_map.total_size_bytes();
        let expected = 8 + 12 + 8 + 5;
        assert_eq!(
            lazy_total, expected,
            "lazy total_size_bytes computes byte_len from metadata"
        );

        // Round-trip through materialize_all → eager TensorMap.
        let eager = lazy_map.materialize_all().expect("materialize");
        assert_eq!(
            eager.total_size_bytes(),
            lazy_total,
            "lazy total_size_bytes must match TensorMap::total_size_bytes \
             post-materialize — iter-3 telemetry swap is byte-identical"
        );
    }

    /// ADR-014 P7 iter-55 — `LazyTensorMap::convert_bf16_to_f16` produces
    /// the same TensorMap (post-materialise) as the eager variant.
    /// Locks the iter-3 swap at main.rs:1188.
    #[test]
    fn test_convert_bf16_to_f16_matches_eager() {
        // Build identical lazy + eager fixtures with mixed BF16/F16/F32 tensors.
        fn bf16_bytes(values: &[f32]) -> Vec<u8> {
            values
                .iter()
                .flat_map(|v| half::bf16::from_f32(*v).to_le_bytes())
                .collect()
        }
        fn f16_bytes(values: &[f32]) -> Vec<u8> {
            values
                .iter()
                .flat_map(|v| half::f16::from_f32(*v).to_le_bytes())
                .collect()
        }

        let f32_a = [1.0_f32, 2.0, 3.0, 4.0];
        let f32_b = [-1.5_f32, 0.5, 2.5];

        // Lazy fixture.
        let mut lazy = LazyTensorMap::new();
        lazy.insert(LazyTensor::from_bytes(
            meta("bf16_a", vec![4], DType::BF16),
            bf16_bytes(&f32_a),
        ));
        lazy.insert(LazyTensor::from_bytes(
            meta("bf16_b", vec![3], DType::BF16),
            bf16_bytes(&f32_b),
        ));
        lazy.insert(LazyTensor::from_bytes(
            meta("f16_c", vec![3], DType::F16),
            f16_bytes(&f32_b),
        ));

        // Eager fixture (same content).
        let mut eager = TensorMap::new();
        eager.insert(TensorRef {
            name: "bf16_a".to_string(),
            shape: vec![4],
            dtype: DType::BF16,
            data: bf16_bytes(&f32_a).into(),
        });
        eager.insert(TensorRef {
            name: "bf16_b".to_string(),
            shape: vec![3],
            dtype: DType::BF16,
            data: bf16_bytes(&f32_b).into(),
        });
        eager.insert(TensorRef {
            name: "f16_c".to_string(),
            shape: vec![3],
            dtype: DType::F16,
            data: f16_bytes(&f32_b).into(),
        });

        let lazy_count = lazy.convert_bf16_to_f16().expect("lazy convert");
        let eager_count = eager.convert_bf16_to_f16().expect("eager convert");
        assert_eq!(lazy_count, eager_count, "convert counts equal");
        assert_eq!(lazy_count, 2, "exactly two BF16 tensors");

        // Materialize lazy, compare bytes pair-by-pair against eager.
        let materialized = lazy.materialize_all().expect("materialize");
        for name in ["bf16_a", "bf16_b", "f16_c"] {
            let m = materialized
                .tensors
                .get(name)
                .unwrap_or_else(|| panic!("materialized missing {name}"));
            let e = eager
                .tensors
                .get(name)
                .unwrap_or_else(|| panic!("eager missing {name}"));
            assert_eq!(
                m.dtype, e.dtype,
                "{name}: dtype must match (BF16 → F16 for converted; F16 unchanged)"
            );
            assert_eq!(
                m.dtype,
                DType::F16,
                "{name}: post-conversion dtype must be F16"
            );
            assert_eq!(
                m.data, e.data,
                "{name}: bytes must equal eager-converted output"
            );
        }
    }

    /// ADR-014 P7 iter-76 — `LazyTensor::from_arc_bytes` materialises
    /// equivalently to `from_bytes`.  When refcount==1 the inner Vec
    /// is unwrapped (zero-copy); when >1 the inner is cloned.
    /// ADR-020 iter-6: `materialize_cloned` Arc-shares (no Vec clone)
    /// for the `MaterializedShared` variant; the `Materialized`
    /// (owned-Vec) variant still wraps a fresh Arc.
    #[test]
    fn test_from_arc_bytes_materialize_byte_equal_to_from_bytes() {
        use std::sync::Arc;

        let m_arc = meta("arc-tensor", vec![3], DType::F32);
        let m_vec = meta("arc-tensor", vec![3], DType::F32);
        let payload = f32_bytes(&[1.0, 2.0, 3.0]);

        // refcount==1 → unwrap path
        let lazy_arc = LazyTensor::from_arc_bytes(m_arc, Arc::new(payload.clone()));
        let lazy_vec = LazyTensor::from_bytes(m_vec, payload.clone());

        let t_arc = lazy_arc.materialize().expect("arc materialize");
        let t_vec = lazy_vec.materialize().expect("vec materialize");

        assert_eq!(*t_arc.data, *t_vec.data, "Arc-path bytes equal Vec-path");
        assert_eq!(t_arc.shape, t_vec.shape);
        assert_eq!(t_arc.dtype, t_vec.dtype);
    }

    #[test]
    fn test_from_arc_bytes_shared_refcount_clones_inner() {
        use std::sync::Arc;

        let payload = f32_bytes(&[7.0, 8.0]);
        let shared: Arc<Vec<u8>> = Arc::new(payload.clone());
        // Hold a second strong ref to force the clone path on materialise.
        let _keepalive = Arc::clone(&shared);

        let lazy = LazyTensor::from_arc_bytes(meta("shared", vec![2], DType::F32), shared);
        let t = lazy.materialize().expect("shared materialize");
        assert_eq!(*t.data, payload, "shared-refcount path returns equal bytes");
    }

    /// ADR-020 iter-6: `materialize_cloned` on `MaterializedShared` shares
    /// the Arc (pointer-bump) instead of deep-cloning the inner Vec.  The
    /// returned `TensorRef.data` and the source Arc must point to the
    /// same allocation (`Arc::ptr_eq`); the lazy retains its own Arc
    /// clone (by-reference semantic preserved); after dropping both
    /// `lazy` and `t`, the source's strong count returns to its pre-call
    /// value.  This is the falsifier for the ~52 GB save on 27B-class
    /// `clone_tensor_map_to_lazy` paths (ADR-020 §8.1 acceptance criteria).
    #[test]
    fn materialize_cloned_shares_arc_no_byte_copy() {
        use std::sync::Arc;

        let payload = f32_bytes(&[1.0, 2.0, 3.0, 4.0]);
        let arc: Arc<Vec<u8>> = Arc::new(payload.clone());
        let pre_count = Arc::strong_count(&arc);
        assert_eq!(pre_count, 1, "test setup: only the test holds the Arc");

        let lazy = LazyTensor::from_arc_bytes(
            meta("share-test", vec![4], DType::F32),
            Arc::clone(&arc),
        );
        // After construction: caller's `arc` + lazy's clone = 2.
        assert_eq!(Arc::strong_count(&arc), 2);

        let t = lazy
            .materialize_cloned()
            .expect("borrowed materialize on shared variant");

        // The returned TensorRef.data must point to the SAME allocation.
        // ptr_eq is the strongest available check — strong_count >= 3
        // is a corollary that proves no byte clone occurred.
        assert!(
            Arc::ptr_eq(&arc, &t.data),
            "materialize_cloned must share the Arc, not deep-clone the Vec"
        );
        assert!(
            Arc::strong_count(&arc) >= 3,
            "Arc::strong_count must be >= 3 (caller + lazy + t); got {}",
            Arc::strong_count(&arc)
        );

        // Bytes still match (read-through deref).
        assert_eq!(**t.data, *payload);

        // Drop t → strong count drops by 1 (back to caller + lazy = 2).
        drop(t);
        assert_eq!(Arc::strong_count(&arc), 2);

        // Drop lazy → strong count back to caller-only.  By-reference
        // semantic verified: lazy never leaked an extra ref past its
        // own lifetime.
        drop(lazy);
        assert_eq!(
            Arc::strong_count(&arc),
            pre_count,
            "post-drop refcount restored — materialize_cloned doesn't leak"
        );
    }

    /// `materialize_cloned` on the `Materialized` (owned-Vec) variant
    /// must still produce a valid `TensorRef` with byte-equal contents.
    /// This variant cannot share an Arc (the Vec is owned, not Arc'd),
    /// so the returned Arc has fresh refcount=1 — that's the one-owner
    /// path documented at lazy.rs module-level docs.
    #[test]
    fn materialize_cloned_owned_vec_path_byte_equal() {
        use std::sync::Arc;

        let payload = f32_bytes(&[5.0, 6.0]);
        let lazy = LazyTensor::from_bytes(meta("owned", vec![2], DType::F32), payload.clone());
        let t = lazy
            .materialize_cloned()
            .expect("borrowed materialize on owned variant");

        assert_eq!(**t.data, *payload);
        assert_eq!(
            Arc::strong_count(&t.data),
            1,
            "owned-Vec variant: returned Arc is fresh (no upstream to share)"
        );
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
        assert_eq!(*a.data, f32_bytes(&[1.0, 2.0]));
        let b = eager.get("b").unwrap();
        assert_eq!(*b.data, f32_bytes(&[42.0]));
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
        assert_eq!(*realised.data, f32_bytes(&[3.14]));
    }
}
