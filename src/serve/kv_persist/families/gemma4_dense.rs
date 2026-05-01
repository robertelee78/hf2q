//! ADR-017 §B-dense.1 — `KvCacheSpill` impl for Gemma 4's dense F32/F16
//! K/V cache (`MlxModelWeights.dense_kvs`).
//!
//! ## What this module owns
//!
//! The byte codec for snapshotting and restoring one `(layer_rank,
//! token_range)` slice of Gemma 4's dense K/V cache. The Phase A.3
//! spiller (`crate::serve::kv_persist::spiller`) owns the on-disk
//! envelope + chain-hash + atomic-rename lifecycle; this module owns
//! the per-block payload format that rides inside the envelope body
//! AND the in-memory cache mutation needed to rehydrate the GPU
//! buffers on a fresh engine admission.
//!
//! ## Activation conditions
//!
//! All three must hold for this hook to do real work (vs. the
//! `Skipped` no-op path):
//!
//! 1. The loaded family is Gemma 4 / GemmaMoE (the operator-side
//!    family resolver registers this hook only for those repos).
//! 2. `HF2Q_KV_PERSIST=on` (Phase C.1's CLI flag wires the spiller
//!    chain at startup; without it the hook never sees a trigger).
//! 3. `HF2Q_USE_DENSE=1` so the engine actually populates `dense_kvs`
//!    at decode (the F32-dense regime). Under TQ-active decode the
//!    hook still receives `post_admit` triggers but its restore is a
//!    no-op because TQ-active layers don't allocate `dense_kvs`.
//!    B-tq.1 is the symmetric hook for the TQ-active codec.
//!
//! ## Cache-shape constants captured at registration
//!
//! Phase C.1 builds the hook from `Gemma4Config` once per loaded
//! `(repo, quant)` pair. The shape (per-layer K/V head count, head
//! dim, layer types, sliding window, max-decode budget, dtype) is
//! immutable for the lifetime of the loaded weights, so capturing it
//! in the hook avoids a re-read of the engine's config on every
//! trigger AND lets the hook short-circuit on shape-mismatch payloads
//! before reaching the GPU.
//!
//! ## Ring-buffer write-pos recovery
//!
//! Sliding layers use a ring buffer keyed on `slot = tok_pos %
//! sliding_window`. Once the ring wraps (token position >=
//! sliding_window), the *physical slot order* no longer encodes the
//! oldest→newest token order — but attention is permutation-invariant
//! over the cached `(K, V)` set under a pure-causal mask, so the
//! restore writes payload bytes into the slot order they occupied
//! pre-eviction. The post-eviction write_pos is captured in
//! `pre_evict` and passed to `post_admit` via the `EngineHandle`'s
//! `Arc<RwLock<Vec<usize>>>` so the next decode step writes the
//! freshly-decoded token at the correct slot index.
//!
//! ## Post-admit-before-prefill allocation
//!
//! At engine admission, `dense_kvs` is `None` — `forward_prefill`
//! allocates it on the first prefill call (lines 274–285 of
//! `forward_prefill.rs`). The spiller's `post_admit` fires BEFORE
//! that first prefill, so this hook's `restore_block` must allocate
//! `dense_kvs` itself (mirroring the `forward_prefill` allocation
//! pattern exactly) before writing payload bytes back. Layers that
//! have no on-disk snapshot stay zero-filled; they'll be correctly
//! populated by the next prefill's per-layer KV write (a layer-by-
//! layer prefill always writes its full slot range; spilled state
//! for a layer is overwritten if-and-only-if the next prefix
//! disagrees, which by definition can't happen for the chain-hashed
//! prefix the spiller guarantees identity-equality on).
//!
//! ## Payload format (binary, little-endian, fixed schema)
//!
//! The payload bytes returned by `snapshot_block` and consumed by
//! `restore_block` are NOT the envelope (the spiller wraps these
//! bytes in an `EnvelopeHeader`); they're a self-describing per-
//! family blob:
//!
//! ```text
//! offset  size  field
//! ------  ----  ----------------------------------------
//!   0      4    magic = b"G4D1" (Gemma 4 dense, version 1)
//!   4      1    dtype tag (0 = F32, 1 = F16)
//!   5      1    is_sliding flag (0 = full-attention, 1 = sliding-ring)
//!   6      2    nkv_heads (u16 LE)
//!   8      2    head_dim   (u16 LE)
//!  10      4    capacity   (u32 LE) — slot count for the layer
//!  14      4    write_pos  (u32 LE) — sliding ring write_pos at
//!                          eviction; u32::MAX for full-attention
//!                          layers (sentinel — full-attn capacity is
//!                          dynamic so write_pos isn't meaningful).
//!  18      4    range_start (u32 LE) — token-position range start
//!  22      4    range_end   (u32 LE) — token-position range end
//!                                       (exclusive)
//!  26      4    k_byte_len  (u32 LE) — count of K bytes that follow
//!  30      4    v_byte_len  (u32 LE) — count of V bytes that follow
//!  34      k_byte_len bytes of K (head-major: nkv_heads × n_slots × head_dim)
//!  ..      v_byte_len bytes of V (same layout as K)
//! ```
//!
//! Both `k_byte_len` and `v_byte_len` equal `nkv_heads * n_slots *
//! head_dim * dtype_bytes` where `n_slots = range.end - range.start`
//! (capped at `capacity`). Header is 34 bytes total; the format is
//! self-describing so the hook can verify shape against its captured
//! config and reject mismatched payloads with `CodecErr` instead of
//! corrupting the cache.
//!
//! Ring-wrap blocks contain the K/V bytes in *token-position* order,
//! not *slot* order — so a restore reading sequentially from token
//! `range.start` to `range.end` gets monotonic content and can write
//! it into ring slots `[start..end] % capacity` without further
//! stitching. This matches how a contiguous decode trajectory writes
//! the cache in the first place.

use std::ops::Range;
use std::sync::{Arc, Mutex, RwLock};

use mlx_native::{DType, MlxBuffer, MlxDevice};

use crate::serve::config::LayerType;
use crate::serve::kv_persist::format::BLOCK_TOKENS;
use crate::serve::kv_persist::spiller::KvCacheSpill;
use crate::serve::multi_model::SpillErrorKind;

/// 4-byte magic at the head of every payload — `b"G4D1"`. Lets the
/// hook reject cross-family payloads (e.g. a B-tq blob accidentally
/// surfaced via stale registry state) before any further parse.
pub const PAYLOAD_MAGIC: &[u8; 4] = b"G4D1";

/// Fixed-size header byte count (sum of fields above).
pub const PAYLOAD_HEADER_BYTES: usize = 34;

/// Sentinel `write_pos` used for full-attention (linear-cap) layers.
/// `u32::MAX` is unrepresentable as a real ring-write position
/// because `capacity <= MAX_POS_EMBED << u32::MAX` for every Gemma 4
/// configuration; reading this back signals "this layer was linear,
/// not ring".
const WRITE_POS_LINEAR_SENTINEL: u32 = u32::MAX;

/// Dtype tag stored in the payload header. Two values today; future
/// quant codecs (BF16, int8) extend this enum and bump the magic.
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum DtypeTag {
    F32 = 0,
    F16 = 1,
}

impl DtypeTag {
    fn from_dtype(dt: DType) -> Result<Self, SpillErrorKind> {
        match dt {
            DType::F32 => Ok(DtypeTag::F32),
            DType::F16 => Ok(DtypeTag::F16),
            // BF16 / U8 / U16 / U32 / I32 are not valid dense KV
            // dtypes for Gemma 4. The forward_prefill allocator only
            // ever picks F32 or F16 (line 260: `kv_dtype = if
            // use_f16_kv { F16 } else { F32 }`).
            _ => Err(SpillErrorKind::CodecErr),
        }
    }

    fn to_dtype(self) -> DType {
        match self {
            DtypeTag::F32 => DType::F32,
            DtypeTag::F16 => DType::F16,
        }
    }

    fn from_byte(b: u8) -> Result<Self, SpillErrorKind> {
        match b {
            0 => Ok(DtypeTag::F32),
            1 => Ok(DtypeTag::F16),
            _ => Err(SpillErrorKind::CodecErr),
        }
    }
}

/// Per-layer dense K/V buffer pair plus shape metadata. Mirrors
/// `crate::serve::forward_mlx::DenseKvBuffers` but lives here to keep
/// the hook independent of the engine's internal types — the C.1
/// wire-up will adapt via `From<&DenseKvBuffers>`.
///
/// Shape is `[nkv_heads, capacity, head_dim]`; element type is `dtype`.
#[derive(Debug)]
pub struct DenseKvBuffer {
    pub k: MlxBuffer,
    pub v: MlxBuffer,
    pub capacity: usize,
    pub is_sliding: bool,
}

/// Live engine handle. C.1 constructs this once per engine admission
/// and calls `set_engine_handle` on the registered hook. The hook
/// reads the device + dense_kvs through the handle on every
/// snapshot/restore call.
///
/// Held behind `Arc<RwLock<Option<EngineHandle>>>` inside the hook so
/// the trait's `&self` / `&mut self` dichotomy on snapshot vs. restore
/// can both reach the underlying state. The handle is replaced
/// (Some → None → Some on re-load) at engine admission/eviction
/// boundaries; stale handles never leak past `clear_engine_handle`.
///
/// `Clone` shares the underlying `MlxDevice` via `Arc`, the dense_kvs
/// slot via `Arc<RwLock<...>>`, and the write_positions via
/// `Arc<RwLock<...>>` — so all clones see the same live cache state.
#[derive(Clone)]
pub struct EngineHandle {
    /// Device used to allocate fresh K/V buffers on a restore-before-
    /// prefill admission. `MlxDevice` is `Send + Sync` but does not
    /// implement `Clone`, so we wrap in `Arc` to keep `EngineHandle`
    /// cheaply cloneable.
    pub device: Arc<MlxDevice>,
    /// Live dense_kvs slot. `None` before the first prefill (or
    /// before the first restore-side allocation). `Some` once any
    /// path has populated it.
    pub dense_kvs: Arc<RwLock<Option<Vec<DenseKvBuffer>>>>,
    /// Per-layer write_pos for ring-buffer recovery. Length equals
    /// `num_layers`; each element is the next slot index a sliding
    /// layer would write at decode (0..capacity). Full-attention
    /// layers store the sentinel `WRITE_POS_LINEAR_SENTINEL as
    /// usize` (cast back to u32 in the payload header).
    pub write_positions: Arc<RwLock<Vec<usize>>>,
}

impl std::fmt::Debug for EngineHandle {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("EngineHandle")
            .field("device", &self.device)
            .field(
                "dense_kvs_present",
                &self
                    .dense_kvs
                    .read()
                    .map(|g| g.is_some())
                    .unwrap_or(false),
            )
            .field(
                "write_positions_len",
                &self.write_positions.read().map(|g| g.len()).unwrap_or(0),
            )
            .finish()
    }
}

/// Phase B-dense.1 `KvCacheSpill` impl for Gemma 4 dense F32/F16 K/V.
///
/// Construction is shape-only (no live engine state); the live engine
/// is wired in via [`Self::set_engine_handle`] at admission time. This
/// matches the spiller's registration model: the hook is registered
/// once per `(repo, quant)` and persists across evictions, but the
/// underlying `LoadedEngine` instance changes on every re-admit.
pub struct Gemma4DenseSpill {
    /// Live engine handle. `None` between evict and admit cycles.
    /// `RwLock` so concurrent snapshot calls from the spiller's
    /// trigger sites don't serialize on a `Mutex`.
    engine: Arc<RwLock<Option<EngineHandle>>>,

    /// Per-layer attention type. Captured from `Gemma4Config` at
    /// registration. Length == num_layers.
    layer_types: Arc<Vec<LayerType>>,

    /// Per-layer KV-head count. For Gemma 4 sliding layers this is
    /// `num_key_value_heads`; for full-attention layers it's
    /// `num_global_key_value_heads`.
    nkv_heads: Arc<Vec<usize>>,

    /// Per-layer head dim. For Gemma 4 sliding layers this is
    /// `head_dim` (256); for full-attention layers it's
    /// `global_head_dim` (512).
    head_dim: Arc<Vec<usize>>,

    /// Shared dtype across all K/V layers. Gemma 4 uses one dtype
    /// for the whole cache: F32 by default, F16 under HF2Q_F16_KV=1.
    kv_dtype: DType,

    /// Sliding-window size. Drives ring-buffer capacity for sliding
    /// layers (`capacity == sliding_window`).
    sliding_window: usize,

    /// Max decode tokens budget. Drives linear capacity for full-
    /// attention layers (`capacity = seq_len + max_decode_tokens`,
    /// computed at restore time using the live prefill seq_len).
    /// Captured here so the restore allocator can reproduce the
    /// `forward_prefill` shape exactly.
    max_decode_tokens: usize,

    /// Number of attention layers. Equals `layer_types.len() ==
    /// nkv_heads.len() == head_dim.len()`.
    num_layers: usize,
}

/// Configuration bundle passed to [`Gemma4DenseSpill::new`]. All
/// fields are captured by value at registration; the hook owns its
/// own `Arc`-wrapped copies after construction.
#[derive(Debug, Clone)]
pub struct Gemma4DenseConfig {
    pub layer_types: Vec<LayerType>,
    pub nkv_heads: Vec<usize>,
    pub head_dim: Vec<usize>,
    pub kv_dtype: DType,
    pub sliding_window: usize,
    pub max_decode_tokens: usize,
}

impl Gemma4DenseSpill {
    /// Build a hook from shape-only config. No live engine state is
    /// captured here — call [`Self::set_engine_handle`] at admission
    /// time before the first `pre_evict` / `post_admit` trigger.
    ///
    /// Returns an error iff the shape vectors disagree on length
    /// (defensive — every C.1 caller will pull these from a single
    /// `Gemma4Config` so they should always agree).
    pub fn new(cfg: Gemma4DenseConfig) -> Result<Self, SpillErrorKind> {
        if cfg.layer_types.len() != cfg.nkv_heads.len()
            || cfg.layer_types.len() != cfg.head_dim.len()
        {
            return Err(SpillErrorKind::CodecErr);
        }
        let num_layers = cfg.layer_types.len();
        // Validate dtype is one we can encode without further work.
        let _tag = DtypeTag::from_dtype(cfg.kv_dtype)?;
        Ok(Self {
            engine: Arc::new(RwLock::new(None)),
            layer_types: Arc::new(cfg.layer_types),
            nkv_heads: Arc::new(cfg.nkv_heads),
            head_dim: Arc::new(cfg.head_dim),
            kv_dtype: cfg.kv_dtype,
            sliding_window: cfg.sliding_window,
            max_decode_tokens: cfg.max_decode_tokens,
            num_layers,
        })
    }

    /// Wire a live engine handle. Called by Phase C.1's `cmd_serve`
    /// at engine-admit time, BEFORE the spiller's `post_admit` fires.
    /// Subsequent calls overwrite the prior handle (the freshest
    /// admission wins).
    pub fn set_engine_handle(&self, handle: EngineHandle) {
        let mut g = self
            .engine
            .write()
            .expect("Gemma4DenseSpill::engine RwLock poisoned");
        *g = Some(handle);
    }

    /// Drop the engine handle (e.g. after eviction). Subsequent
    /// `snapshot_block` calls return `None` (Skipped) until the next
    /// `set_engine_handle`.
    pub fn clear_engine_handle(&self) {
        let mut g = self
            .engine
            .write()
            .expect("Gemma4DenseSpill::engine RwLock poisoned");
        *g = None;
    }

    /// Shape introspection: how many layers this hook manages.
    pub fn num_layers(&self) -> usize {
        self.num_layers
    }

    /// Shape introspection: dtype of K/V elements.
    pub fn kv_dtype(&self) -> DType {
        self.kv_dtype
    }

    /// Per-layer capacity (slot count) — matches the
    /// `forward_prefill.rs:274–285` allocator. Sliding layers use
    /// `sliding_window`; full-attention layers use a linear capacity
    /// derived from `seq_len + max_decode_tokens`. The seq_len is
    /// not known at registration; the restore allocator passes it in.
    fn capacity_for_layer(&self, layer_idx: usize, seq_len: usize) -> usize {
        match self.layer_types[layer_idx] {
            LayerType::Sliding => self.sliding_window,
            LayerType::Full => seq_len.saturating_add(self.max_decode_tokens),
        }
    }

    /// Allocate one layer's `DenseKvBuffer` mirroring the prefill
    /// allocator (`forward_prefill.rs:274–285`). Returns
    /// `SpillErrorKind::IoErr` on Metal allocation failure (rare;
    /// surfaces as OOM at the GPU level).
    fn alloc_layer(
        &self,
        device: &MlxDevice,
        layer_idx: usize,
        seq_len: usize,
    ) -> Result<DenseKvBuffer, SpillErrorKind> {
        let nkv = self.nkv_heads[layer_idx];
        let hd = self.head_dim[layer_idx];
        let is_sliding = self.layer_types[layer_idx] == LayerType::Sliding;
        let capacity = self.capacity_for_layer(layer_idx, seq_len);
        let n = nkv * capacity * hd;
        let elem_bytes = self.kv_dtype.size_of();
        let byte_len = n * elem_bytes;
        let k = device
            .alloc_buffer(byte_len, self.kv_dtype, vec![nkv, capacity, hd])
            .map_err(|_| SpillErrorKind::IoErr)?;
        let v = device
            .alloc_buffer(byte_len, self.kv_dtype, vec![nkv, capacity, hd])
            .map_err(|_| SpillErrorKind::IoErr)?;
        Ok(DenseKvBuffer {
            k,
            v,
            capacity,
            is_sliding,
        })
    }

    /// Allocate every layer, mirroring `forward_prefill.rs:274–285`'s
    /// loop. `seq_len_hint` drives the full-attention linear capacity;
    /// for the restore-before-prefill case we pass `0` so capacity
    /// equals `max_decode_tokens` (the next prefill will reallocate
    /// at its real seq_len when we drop this allocation — but as long
    /// as the spilled tokens fit, restore stays correct).
    ///
    /// The restore-allocate-then-prefill-reallocate pattern is the
    /// price of `post_admit` firing before the engine knows its next
    /// prefix length. The next-prefill reallocation is unconditional
    /// in `forward_prefill.rs:274–285` (`Vec::with_capacity` resets
    /// `dense_kvs`); our pre-allocated buffers are dropped at that
    /// point. To avoid the realloc churn, the restore writes content
    /// directly into where the prefill allocator would write — but
    /// since we don't know the prefill's seq_len in advance, the
    /// safest correctness-first choice is "allocate now, let prefill
    /// realloc and re-populate". B-dense.2's parity matrix locks in
    /// that the realloc + repopulate cycle still produces byte-exact
    /// decode tokens.
    ///
    /// For sliding layers the capacity is invariant under seq_len so
    /// restore-allocated buffers survive prefill unchanged.
    fn alloc_all_layers(
        &self,
        device: &MlxDevice,
        seq_len_hint: usize,
    ) -> Result<Vec<DenseKvBuffer>, SpillErrorKind> {
        let mut out = Vec::with_capacity(self.num_layers);
        for li in 0..self.num_layers {
            out.push(self.alloc_layer(device, li, seq_len_hint)?);
        }
        Ok(out)
    }

    /// Convert the per-layer write_pos (or sentinel) to its on-disk
    /// u32 representation.
    fn pack_write_pos(&self, layer_idx: usize, raw: usize) -> u32 {
        match self.layer_types[layer_idx] {
            LayerType::Sliding => {
                // Defensively cap at capacity to avoid surprises if
                // the engine over-shoots. Real sliding layers always
                // hold `write_pos < capacity`.
                let cap = self.sliding_window as u32;
                let v = (raw as u32).min(cap.saturating_sub(1));
                if cap == 0 {
                    0
                } else {
                    v
                }
            }
            LayerType::Full => WRITE_POS_LINEAR_SENTINEL,
        }
    }

    /// Validate decoded payload-header fields against this hook's
    /// captured config. Returns `CodecErr` on any mismatch.
    fn validate_header(
        &self,
        layer_rank: usize,
        hdr: &PayloadHeader,
    ) -> Result<(), SpillErrorKind> {
        if layer_rank >= self.num_layers {
            return Err(SpillErrorKind::CodecErr);
        }
        if hdr.dtype.to_dtype() != self.kv_dtype {
            return Err(SpillErrorKind::CodecErr);
        }
        let want_sliding = self.layer_types[layer_rank] == LayerType::Sliding;
        if hdr.is_sliding != want_sliding {
            return Err(SpillErrorKind::CodecErr);
        }
        if hdr.nkv_heads as usize != self.nkv_heads[layer_rank] {
            return Err(SpillErrorKind::CodecErr);
        }
        if hdr.head_dim as usize != self.head_dim[layer_rank] {
            return Err(SpillErrorKind::CodecErr);
        }
        if hdr.range_start > hdr.range_end {
            return Err(SpillErrorKind::CodecErr);
        }
        Ok(())
    }
}

/// Decoded payload-header view. `from_bytes` parses the on-disk
/// 34-byte header; `encode_into` writes it back. Both fail on
/// malformed input via `CodecErr`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct PayloadHeader {
    dtype: DtypeTag,
    is_sliding: bool,
    nkv_heads: u16,
    head_dim: u16,
    capacity: u32,
    write_pos: u32,
    range_start: u32,
    range_end: u32,
    k_byte_len: u32,
    v_byte_len: u32,
}

impl PayloadHeader {
    fn encode_into(&self, out: &mut Vec<u8>) {
        out.extend_from_slice(PAYLOAD_MAGIC);
        out.push(self.dtype as u8);
        out.push(if self.is_sliding { 1 } else { 0 });
        out.extend_from_slice(&self.nkv_heads.to_le_bytes());
        out.extend_from_slice(&self.head_dim.to_le_bytes());
        out.extend_from_slice(&self.capacity.to_le_bytes());
        out.extend_from_slice(&self.write_pos.to_le_bytes());
        out.extend_from_slice(&self.range_start.to_le_bytes());
        out.extend_from_slice(&self.range_end.to_le_bytes());
        out.extend_from_slice(&self.k_byte_len.to_le_bytes());
        out.extend_from_slice(&self.v_byte_len.to_le_bytes());
    }

    fn from_bytes(buf: &[u8]) -> Result<Self, SpillErrorKind> {
        if buf.len() < PAYLOAD_HEADER_BYTES {
            return Err(SpillErrorKind::CodecErr);
        }
        if &buf[0..4] != PAYLOAD_MAGIC.as_slice() {
            return Err(SpillErrorKind::CodecErr);
        }
        let dtype = DtypeTag::from_byte(buf[4])?;
        let is_sliding = match buf[5] {
            0 => false,
            1 => true,
            _ => return Err(SpillErrorKind::CodecErr),
        };
        let nkv_heads = u16::from_le_bytes([buf[6], buf[7]]);
        let head_dim = u16::from_le_bytes([buf[8], buf[9]]);
        let capacity = u32::from_le_bytes([buf[10], buf[11], buf[12], buf[13]]);
        let write_pos = u32::from_le_bytes([buf[14], buf[15], buf[16], buf[17]]);
        let range_start = u32::from_le_bytes([buf[18], buf[19], buf[20], buf[21]]);
        let range_end = u32::from_le_bytes([buf[22], buf[23], buf[24], buf[25]]);
        let k_byte_len = u32::from_le_bytes([buf[26], buf[27], buf[28], buf[29]]);
        let v_byte_len = u32::from_le_bytes([buf[30], buf[31], buf[32], buf[33]]);
        Ok(Self {
            dtype,
            is_sliding,
            nkv_heads,
            head_dim,
            capacity,
            write_pos,
            range_start,
            range_end,
            k_byte_len,
            v_byte_len,
        })
    }
}

/// Compute the per-token byte stride for a `(nkv_heads, head_dim,
/// dtype)` triple. One token's K (or V) state spans
/// `nkv_heads * head_dim * dtype_bytes` bytes.
fn token_stride_bytes(nkv_heads: usize, head_dim: usize, dtype: DType) -> usize {
    nkv_heads * head_dim * dtype.size_of()
}

/// Read a contiguous run of tokens from a head-major K (or V) buffer
/// in *token-position* order. For sliding (ring) layers the run may
/// wrap the ring boundary — in that case we emit two segments
/// concatenated in token order (first segment from
/// `[start_slot..capacity)`, second from `[0..end_slot)`).
///
/// The buffer layout is `[nkv_heads, capacity, head_dim]` row-major,
/// so token position `tok` of head `h` lives at byte offset
/// `(h * capacity + (tok % capacity)) * head_dim * dtype_bytes`.
/// Reading "token tok across all heads" requires gathering one
/// `head_dim * dtype_bytes` chunk from each head — we do this by
/// looping heads on the outside and tokens on the inside.
///
/// Output layout matches the buffer: `[nkv_heads, n_tokens,
/// head_dim]` head-major. So for a contiguous range we can copy
/// `nkv_heads` strided runs directly.
fn read_kv_range_to_bytes(
    src: &MlxBuffer,
    nkv_heads: usize,
    capacity: usize,
    head_dim: usize,
    dtype: DType,
    is_sliding: bool,
    range: Range<u32>,
) -> Result<Vec<u8>, SpillErrorKind> {
    let n_tokens = (range.end - range.start) as usize;
    let elem_bytes = dtype.size_of();
    let head_stride_bytes = capacity * head_dim * elem_bytes;
    let tok_chunk_bytes = head_dim * elem_bytes;

    // SAFETY contract on as_slice<u8>: caller upholds "no GPU command
    // buffer that writes this buffer is currently in flight". The
    // spiller's eviction trigger fires from `HotSwapManager::evict` /
    // `load_or_get` after the engine's prefill+decode loop has
    // returned to idle, so all GPU work is drained.
    let src_bytes: &[u8] = src
        .as_slice::<u8>()
        .map_err(|_| SpillErrorKind::CodecErr)?;
    let expected_bytes = nkv_heads * capacity * head_dim * elem_bytes;
    if src_bytes.len() < expected_bytes {
        return Err(SpillErrorKind::CodecErr);
    }

    let mut out = Vec::with_capacity(nkv_heads * n_tokens * tok_chunk_bytes);

    if is_sliding {
        // Ring-buffer read: emit tokens in token-position order.
        // For each head, loop the tokens [range.start..range.end]
        // and read `slot = tok % capacity`. Two contiguous segments
        // when the range wraps; one when it doesn't.
        if capacity == 0 {
            return Err(SpillErrorKind::CodecErr);
        }
        for h in 0..nkv_heads {
            let head_base = h * head_stride_bytes;
            for tok in range.start..range.end {
                let slot = (tok as usize) % capacity;
                let off = head_base + slot * tok_chunk_bytes;
                let end = off + tok_chunk_bytes;
                if end > src_bytes.len() {
                    return Err(SpillErrorKind::CodecErr);
                }
                out.extend_from_slice(&src_bytes[off..end]);
            }
        }
    } else {
        // Linear (full-attention) layer: tokens are stored at slot
        // == position. The spiller's caller bounds the range to the
        // populated prefix, so we trust that here. A monotonic loop
        // over heads × tokens emits the same shape as the ring path.
        for h in 0..nkv_heads {
            let head_base = h * head_stride_bytes;
            for tok in range.start..range.end {
                let slot = tok as usize;
                if slot >= capacity {
                    return Err(SpillErrorKind::CodecErr);
                }
                let off = head_base + slot * tok_chunk_bytes;
                let end = off + tok_chunk_bytes;
                if end > src_bytes.len() {
                    return Err(SpillErrorKind::CodecErr);
                }
                out.extend_from_slice(&src_bytes[off..end]);
            }
        }
    }

    Ok(out)
}

/// Write a contiguous run of tokens into a head-major K (or V)
/// buffer in *token-position* order — the inverse of
/// [`read_kv_range_to_bytes`]. Sliding layers write into ring slots
/// `[start..end] % capacity`; full-attention layers write into
/// `slot == position`.
fn write_bytes_into_kv_range(
    dst: &mut MlxBuffer,
    nkv_heads: usize,
    capacity: usize,
    head_dim: usize,
    dtype: DType,
    is_sliding: bool,
    range: Range<u32>,
    payload: &[u8],
) -> Result<(), SpillErrorKind> {
    let n_tokens = (range.end - range.start) as usize;
    let elem_bytes = dtype.size_of();
    let head_stride_bytes = capacity * head_dim * elem_bytes;
    let tok_chunk_bytes = head_dim * elem_bytes;
    let expected_payload = nkv_heads * n_tokens * tok_chunk_bytes;
    if payload.len() != expected_payload {
        return Err(SpillErrorKind::CodecErr);
    }
    let dst_bytes: &mut [u8] = dst
        .as_mut_slice::<u8>()
        .map_err(|_| SpillErrorKind::CodecErr)?;
    let dst_total = nkv_heads * capacity * head_dim * elem_bytes;
    if dst_bytes.len() < dst_total {
        return Err(SpillErrorKind::CodecErr);
    }

    if is_sliding {
        if capacity == 0 {
            return Err(SpillErrorKind::CodecErr);
        }
        let mut payload_off = 0usize;
        for h in 0..nkv_heads {
            let head_base = h * head_stride_bytes;
            for tok in range.start..range.end {
                let slot = (tok as usize) % capacity;
                let off = head_base + slot * tok_chunk_bytes;
                let end = off + tok_chunk_bytes;
                if end > dst_bytes.len() {
                    return Err(SpillErrorKind::CodecErr);
                }
                dst_bytes[off..end]
                    .copy_from_slice(&payload[payload_off..payload_off + tok_chunk_bytes]);
                payload_off += tok_chunk_bytes;
            }
        }
    } else {
        let mut payload_off = 0usize;
        for h in 0..nkv_heads {
            let head_base = h * head_stride_bytes;
            for tok in range.start..range.end {
                let slot = tok as usize;
                if slot >= capacity {
                    return Err(SpillErrorKind::CodecErr);
                }
                let off = head_base + slot * tok_chunk_bytes;
                let end = off + tok_chunk_bytes;
                if end > dst_bytes.len() {
                    return Err(SpillErrorKind::CodecErr);
                }
                dst_bytes[off..end]
                    .copy_from_slice(&payload[payload_off..payload_off + tok_chunk_bytes]);
                payload_off += tok_chunk_bytes;
            }
        }
    }

    Ok(())
}

impl KvCacheSpill for Gemma4DenseSpill {
    fn block_alignment(&self) -> u32 {
        BLOCK_TOKENS
    }

    fn snapshot_block(&self, layer_rank: usize, range: Range<u32>) -> Option<Vec<u8>> {
        // 1. Bounds + sanity checks.
        if layer_rank >= self.num_layers {
            return None;
        }
        if range.end <= range.start {
            return None;
        }

        // 2. Engine handle present?
        let engine_guard = self.engine.read().ok()?;
        let engine = engine_guard.as_ref()?;

        // 3. Read dense_kvs.
        let kvs_guard = engine.dense_kvs.read().ok()?;
        let kvs = kvs_guard.as_ref()?;
        if layer_rank >= kvs.len() {
            return None;
        }
        let layer = &kvs[layer_rank];

        // 4. Validate layer shape against captured config.
        let nkv = self.nkv_heads[layer_rank];
        let hd = self.head_dim[layer_rank];
        let is_sliding = self.layer_types[layer_rank] == LayerType::Sliding;
        if layer.is_sliding != is_sliding {
            return None;
        }
        let capacity = layer.capacity;
        if capacity == 0 {
            return None;
        }
        // Sliding: range must fit within capacity (a snapshot can't
        // exceed the ring's storage). Full-attn: range must fit within
        // capacity. Both checks collapse to one here.
        let n_tokens = (range.end - range.start) as usize;
        if n_tokens > capacity {
            return None;
        }

        // 5. Read K/V byte ranges from GPU buffer (zero-copy on
        //    StorageModeShared via as_slice<u8>).
        let k_bytes = read_kv_range_to_bytes(
            &layer.k,
            nkv,
            capacity,
            hd,
            self.kv_dtype,
            is_sliding,
            range.clone(),
        )
        .ok()?;
        let v_bytes = read_kv_range_to_bytes(
            &layer.v,
            nkv,
            capacity,
            hd,
            self.kv_dtype,
            is_sliding,
            range.clone(),
        )
        .ok()?;

        // 6. Read write_pos for sliding layers.
        let raw_pos = engine
            .write_positions
            .read()
            .ok()
            .and_then(|g| g.get(layer_rank).copied())
            .unwrap_or(0);
        let packed_pos = self.pack_write_pos(layer_rank, raw_pos);

        // 7. Build payload header + body.
        let dtype_tag = DtypeTag::from_dtype(self.kv_dtype).ok()?;
        let header = PayloadHeader {
            dtype: dtype_tag,
            is_sliding,
            nkv_heads: nkv as u16,
            head_dim: hd as u16,
            capacity: capacity as u32,
            write_pos: packed_pos,
            range_start: range.start,
            range_end: range.end,
            k_byte_len: k_bytes.len() as u32,
            v_byte_len: v_bytes.len() as u32,
        };

        // Sanity: K/V byte counts must match the
        // `nkv * n_tokens * head_dim * dtype_bytes` formula.
        let stride = token_stride_bytes(nkv, hd, self.kv_dtype);
        let expected = nkv * n_tokens * stride / nkv;
        // Above simplifies to n_tokens * stride; written verbosely
        // for reviewer clarity.
        let _ = expected;
        let expected_kv_bytes = nkv * n_tokens * hd * self.kv_dtype.size_of();
        if k_bytes.len() != expected_kv_bytes || v_bytes.len() != expected_kv_bytes {
            return None;
        }

        let mut out = Vec::with_capacity(PAYLOAD_HEADER_BYTES + k_bytes.len() + v_bytes.len());
        header.encode_into(&mut out);
        out.extend_from_slice(&k_bytes);
        out.extend_from_slice(&v_bytes);
        Some(out)
    }

    fn restore_block(
        &mut self,
        layer_rank: usize,
        range: Range<u32>,
        payload: &[u8],
    ) -> Result<(), SpillErrorKind> {
        // 1. Parse header.
        let hdr = PayloadHeader::from_bytes(payload)?;

        // 2. Validate against captured config.
        self.validate_header(layer_rank, &hdr)?;

        // 3. Validate range agrees with payload header. The spiller
        //    passes `0..n_tokens` derived from `EnvelopeHeader.n_tokens`
        //    (A.3 contract), so the payload's own range fields are
        //    the authoritative slot range; we trust them and assert
        //    the spiller's range matches in `n_tokens`.
        let n_tokens_payload = (hdr.range_end - hdr.range_start) as usize;
        let n_tokens_spiller = (range.end - range.start) as usize;
        if n_tokens_payload != n_tokens_spiller {
            return Err(SpillErrorKind::CodecErr);
        }

        // 4. Validate body length.
        let stride = token_stride_bytes(
            hdr.nkv_heads as usize,
            hdr.head_dim as usize,
            hdr.dtype.to_dtype(),
        );
        let expected_kv = n_tokens_payload * stride;
        if hdr.k_byte_len as usize != expected_kv
            || hdr.v_byte_len as usize != expected_kv
        {
            return Err(SpillErrorKind::CodecErr);
        }
        let body_start = PAYLOAD_HEADER_BYTES;
        let body_total = hdr.k_byte_len as usize + hdr.v_byte_len as usize;
        if payload.len() != body_start + body_total {
            return Err(SpillErrorKind::CodecErr);
        }

        // 5. Engine handle present? Without a handle the restore
        //    cannot mutate any cache; surface as CodecErr (the
        //    spiller's contract is "errors come back here", and
        //    a missing handle at restore time is a configuration
        //    bug — Phase C.1 wires the handle BEFORE post_admit
        //    fires).
        let engine_guard = self
            .engine
            .read()
            .map_err(|_| SpillErrorKind::CodecErr)?;
        let engine = engine_guard.as_ref().ok_or(SpillErrorKind::CodecErr)?;

        // 6. Allocate dense_kvs if absent. The spiller fires
        //    post_admit BEFORE the first prefill, at which point the
        //    engine's dense_kvs is still None.
        {
            let mut kvs_guard = engine
                .dense_kvs
                .write()
                .map_err(|_| SpillErrorKind::CodecErr)?;
            if kvs_guard.is_none() {
                let layers = self.alloc_all_layers(&engine.device, 0)?;
                *kvs_guard = Some(layers);
                // Resize write_positions to match if needed.
                let mut wp = engine
                    .write_positions
                    .write()
                    .map_err(|_| SpillErrorKind::CodecErr)?;
                if wp.len() != self.num_layers {
                    wp.clear();
                    wp.resize(self.num_layers, 0);
                }
            }
        }

        // 7. Validate the freshly-allocated (or already-allocated)
        //    layer's capacity matches the payload's recorded
        //    capacity. Sliding capacity is fixed at sliding_window so
        //    must match exactly. Full-attn capacity may differ
        //    (allocator used seq_len_hint=0 → cap = max_decode_tokens)
        //    but only need to be at least as large as the payload's
        //    range_end. CodecErr otherwise.
        let mut kvs_guard = engine
            .dense_kvs
            .write()
            .map_err(|_| SpillErrorKind::CodecErr)?;
        let kvs = kvs_guard.as_mut().ok_or(SpillErrorKind::CodecErr)?;
        if layer_rank >= kvs.len() {
            return Err(SpillErrorKind::CodecErr);
        }
        let layer = &mut kvs[layer_rank];
        let nkv = hdr.nkv_heads as usize;
        let hd = hdr.head_dim as usize;
        let dtype = hdr.dtype.to_dtype();
        if hdr.is_sliding {
            if layer.capacity != hdr.capacity as usize {
                return Err(SpillErrorKind::CodecErr);
            }
        } else if (layer.capacity as u32) < hdr.range_end {
            return Err(SpillErrorKind::CodecErr);
        }

        // 8. Memcpy K + V payload chunks into ring/linear slots.
        let body = &payload[body_start..];
        let k_slice = &body[..hdr.k_byte_len as usize];
        let v_slice = &body[hdr.k_byte_len as usize..];
        write_bytes_into_kv_range(
            &mut layer.k,
            nkv,
            layer.capacity,
            hd,
            dtype,
            hdr.is_sliding,
            range.clone(),
            k_slice,
        )?;
        write_bytes_into_kv_range(
            &mut layer.v,
            nkv,
            layer.capacity,
            hd,
            dtype,
            hdr.is_sliding,
            range.clone(),
            v_slice,
        )?;

        // 9. Restore write_pos for sliding layers.
        if hdr.is_sliding && hdr.write_pos != WRITE_POS_LINEAR_SENTINEL {
            let mut wp = engine
                .write_positions
                .write()
                .map_err(|_| SpillErrorKind::CodecErr)?;
            if layer_rank < wp.len() {
                wp[layer_rank] = hdr.write_pos as usize;
            }
        }

        Ok(())
    }
}

// Send + Sync invariant: every field is Arc-wrapped Send+Sync state.
// `MlxDevice` (inside EngineHandle) is documented Send+Sync via
// metal::Device + Arc; `MlxBuffer` is Send+Sync per
// `mlx-native/src/buffer.rs:80` `static_assertions_send_sync!`.
// Compile-time witnesses below.
#[doc(hidden)]
#[allow(dead_code)]
fn _assert_send_sync() {
    fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<Gemma4DenseSpill>();
    assert_send_sync::<EngineHandle>();
    // Mutex<Gemma4DenseSpill> wraps the spill hook in
    // `Arc<Mutex<dyn KvCacheSpill>>` per spiller.rs's FamilyHook
    // type alias — this static check makes sure the trait object
    // itself is Send.
    assert_send_sync::<Arc<Mutex<dyn KvCacheSpill>>>();
}

// ---------------------------------------------------------------------------
// Phase C.1 — `EngineBindable` impl for `Gemma4DenseSpill`.
//
// The C.1 `LoaderWrapper` delivers a type-erased `Arc<dyn Any + Send + Sync>`
// at load time. For Gemma 4 the expected concrete is `Arc<EngineHandle>`
// — the live device + dense_kvs slot + write_positions tuple defined
// above. The impl downcasts via `Arc::downcast::<EngineHandle>()`:
//
//   * On success → clone the inner EngineHandle (cheap-clone via the
//     existing `EngineHandle::clone` impl above) and store via the
//     existing `set_engine_handle` method (which acquires the
//     interior `Arc<RwLock<Option<EngineHandle>>>`'s write lock).
//   * On mismatch → silently no-op. The downcast returns
//     `Err(Arc<dyn Any>)` which we drop at end of scope. This
//     satisfies the contract: hooks NEVER panic on type mismatch.
//
// `unbind_engine` clears the engine handle slot via the existing
// `clear_engine_handle` method. Subsequent `snapshot_block` calls
// then return `None` (the existing engine-handle-None short-circuit
// at `gemma4_dense.rs:743-744`).
//
// The C.1 cmd_serve wire-up DOES NOT route the loaded `Engine` here
// directly — instead, cmd_serve builds an `Arc<EngineHandle>` from
// the engine's MlxModelWeights once load_or_get returns and binds
// THAT through the registry. The LoaderWrapper's automatic bind path
// (which would deliver `Arc<Engine>` here) finds the type mismatch
// and silently no-ops; the explicit `cmd_serve` bind delivers the
// `Arc<EngineHandle>` and succeeds. This split lets the Engine
// remain fully owned by the manager (no Arc retention contract
// violation) while still threading the live KV state to the hook.
// ---------------------------------------------------------------------------

impl crate::serve::kv_persist::EngineBindable for Gemma4DenseSpill {
    fn bind_engine(&self, engine_dyn: Arc<dyn std::any::Any + Send + Sync>) {
        // Try downcast to Arc<EngineHandle>. On Err, the type-erased
        // Arc is returned to us; drop it without further action.
        match engine_dyn.downcast::<EngineHandle>() {
            Ok(handle_arc) => {
                // Cheap-clone the inner EngineHandle (every field is
                // `Arc`-wrapped). The downcast Arc itself drops at
                // end of scope.
                let handle: EngineHandle = (*handle_arc).clone();
                self.set_engine_handle(handle);
            }
            Err(_other_dyn) => {
                // Silent no-op on type mismatch (per the
                // EngineBindable contract). The wrapped Any drops
                // at end of scope.
            }
        }
    }

    fn unbind_engine(&self) {
        self.clear_engine_handle();
    }
}

// ---------------------------------------------------------------------------
// Phase B-dense.2 — `FamilyHookFactory` impl for `Gemma4DenseSpill`.
//
// The C.1 wire-up registers a *stub* `EngineBindable` at `cmd_serve`
// startup because the real `Gemma4DenseSpill` shape config comes from
// the live engine (post-load `MlxModelWeights`). B-dense.2 closes that
// gap with a factory: at startup, `cmd_serve` registers a
// `Gemma4DenseSpillFactory` carrying the shape config; on the first
// successful load, the registry's `try_substitute_on_load` invokes
// `factory.try_construct(engine_dyn)` and atomically substitutes the
// stub for a real `Gemma4DenseSpill` already wired to the engine.
//
// ## Construction-from-engine helper (`try_from_engine_arc`)
//
// `Gemma4DenseSpill::new(cfg)` is shape-only. The factory's job is
// to invoke `set_engine_handle` on the freshly-constructed spill IFF
// the type-erased engine downcasts to `Arc<EngineHandle>`. The
// helper returns `Option<Self>` (None on type mismatch); the
// factory's `try_construct` wraps this in the larger
// `(kv_hook, bindable_hook)` tuple.
//
// ## Why downcast to `Arc<EngineHandle>` (not `Arc<Engine>`)
//
// Phase C.1's `LoaderWrapper` auto-delivers `Arc<Engine>` after every
// load — but the `Engine` type is owned by the worker thread (it's
// `Clone` over `Arc<EngineInner>`); pulling shape config out of it
// would cross the worker-thread boundary. The cleanly-decoupled
// design is: `cmd_serve` builds an `Arc<EngineHandle>` directly from
// the loaded `MlxModelWeights` (already done in C.1 for the
// `EngineBindable::bind_engine` path) and feeds that
// `Arc<EngineHandle>` through the registry's
// `try_substitute_on_load`. The auto-path's `Arc<Engine>` mismatches
// our downcast and returns `None` cleanly.
//
// ## Why no panic on type mismatch
//
// `Arc::downcast::<EngineHandle>()` returns `Result<Arc<EngineHandle>, Arc<dyn Any>>`
// — the `Err` branch drops the type-erased Arc silently. This
// mirrors the existing `EngineBindable::bind_engine` impl above and
// satisfies the `FamilyHookFactory` contract (no panic on type
// mismatch).
// ---------------------------------------------------------------------------

impl Gemma4DenseSpill {
    /// Phase B-dense.2 — try to construct a `Gemma4DenseSpill` whose
    /// engine handle is already wired. Returns `Some(spill)` iff
    /// `engine_dyn` downcasts to `Arc<EngineHandle>`; returns
    /// `None` on type mismatch (the type-erased Arc drops silently).
    ///
    /// `cfg` carries the shape-only config (per
    /// [`Gemma4DenseConfig`]); the engine handle delivers the live
    /// device + dense_kvs slot + write_positions tuple.
    ///
    /// Returns `None` when:
    ///   * `engine_dyn` is not `Arc<EngineHandle>`, OR
    ///   * `Gemma4DenseSpill::new(cfg)` fails (shape vector length
    ///     mismatch — defensive; shouldn't fire under
    ///     well-formed cmd_serve registration).
    pub fn try_from_engine_arc(
        cfg: Gemma4DenseConfig,
        engine_dyn: Arc<dyn std::any::Any + Send + Sync>,
    ) -> Option<Self> {
        let handle_arc = match engine_dyn.downcast::<EngineHandle>() {
            Ok(arc) => arc,
            Err(_other_dyn) => return None,
        };
        let spill = match Self::new(cfg) {
            Ok(s) => s,
            Err(_) => return None,
        };
        // Cheap-clone the inner EngineHandle (every field is
        // Arc-wrapped). The downcast Arc itself drops at end of scope.
        let handle: EngineHandle = (*handle_arc).clone();
        spill.set_engine_handle(handle);
        Some(spill)
    }
}

/// Phase B-dense.2 — `FamilyHookFactory` impl for `Gemma4DenseSpill`.
///
/// Carries the shape-only `Gemma4DenseConfig` captured at `cmd_serve`
/// startup. On `try_construct(engine_dyn)`:
///
///   1. Calls [`Gemma4DenseSpill::try_from_engine_arc`] which
///      downcasts to `Arc<EngineHandle>` and constructs a fully-wired
///      spill on success.
///   2. On `None` (type mismatch) → returns `None` (the registry's
///      caller leaves the prior stub registration in place).
///   3. On `Some(spill)` → wraps the spill in the
///      `(Arc<Mutex<dyn KvCacheSpill>>, Arc<dyn EngineBindable>)`
///      tuple expected by the `FamilyHookFactory` trait. The same
///      `Arc<Gemma4DenseSpill>` is referenced by both ends of the
///      tuple — so a `bind_engine` call through the registry side
///      mutates the same engine slot that a `restore_block` call
///      through the spiller side reads.
pub struct Gemma4DenseSpillFactory {
    cfg: Gemma4DenseConfig,
}

impl Gemma4DenseSpillFactory {
    /// Construct a factory carrying the supplied shape config. The
    /// config is captured by value; subsequent factory invocations
    /// reuse the same shape (the loaded model's shape is immutable
    /// across evict/readmit cycles).
    pub fn new(cfg: Gemma4DenseConfig) -> Self {
        Self { cfg }
    }
}

impl crate::serve::kv_persist::registry::FamilyHookFactory for Gemma4DenseSpillFactory {
    fn try_construct(
        &self,
        engine_dyn: Arc<dyn std::any::Any + Send + Sync>,
    ) -> Option<(
        Arc<Mutex<dyn crate::serve::kv_persist::spiller::KvCacheSpill>>,
        Arc<dyn crate::serve::kv_persist::EngineBindable>,
    )> {
        // 1. Try to materialize a fully-wired spill from the engine.
        //    None on type mismatch ⇒ no substitution.
        let spill = Gemma4DenseSpill::try_from_engine_arc(self.cfg.clone(), engine_dyn)?;
        // 2. Same Arc<Gemma4DenseSpill> is referenced by both ends of
        //    the tuple. The kv_hook side wraps in Mutex (per
        //    KvCacheSpill::restore_block's &mut self contract); the
        //    bindable side is a direct Arc<dyn EngineBindable>.
        //
        //    Note: a Mutex around the spill on the kv side does NOT
        //    serialize against the registry side — bind_engine /
        //    unbind_engine on the spill take the inner
        //    `engine: Arc<RwLock<Option<EngineHandle>>>` lock directly,
        //    not the outer Mutex. So a long-running restore_block
        //    holding the outer Mutex doesn't block a bind/unbind.
        let spill_arc = Arc::new(spill);
        let bindable: Arc<dyn crate::serve::kv_persist::EngineBindable> = spill_arc.clone();
        // The Mutex<dyn KvCacheSpill> tuple side wraps a fresh
        // Gemma4DenseSpill clone — but Gemma4DenseSpill is NOT
        // `Clone`. The standard pattern (mirrors how cmd_serve C.1
        // splits the stub: `let stub = Arc::new(StubGemma4Spill); let
        // stub_for_spiller: Arc<Mutex<dyn KvCacheSpill>> = Arc::new(Mutex::new(StubGemma4Spill));`)
        // is to construct a *second* spill instance for the spiller
        // side. The two instances share the same engine state through
        // their inner `Arc<RwLock<Option<EngineHandle>>>` ONLY IF we
        // bind both. To keep the two synchronized, we re-run
        // try_from_engine_arc with a fresh clone of the type-erased
        // engine — but we no longer have it after the move into the
        // first call. Instead, we materialize the spiller-side spill
        // from cfg + handle pulled directly from the bindable side's
        // engine slot.
        //
        // Architecturally simpler approach: build TWO independent
        // spill instances, each `set_engine_handle`'d from a clone of
        // the EngineHandle. Both reach the same `dense_kvs:
        // Arc<RwLock<Option<Vec<DenseKvBuffer>>>>` because EngineHandle
        // is `Clone` and every field is `Arc`-wrapped.
        let handle_for_spiller: EngineHandle = {
            let g = spill_arc
                .engine
                .read()
                .expect("Gemma4DenseSpill::engine RwLock poisoned");
            // We just constructed + set_engine_handle'd this; it must
            // be Some at this point.
            match g.as_ref() {
                Some(h) => h.clone(),
                None => {
                    // Defensive: if for some reason set_engine_handle
                    // didn't take (lock poisoned, etc.), fall back to
                    // returning None — the spiller side is unusable
                    // without a handle.
                    return None;
                }
            }
        };
        let spiller_side = match Gemma4DenseSpill::new(self.cfg.clone()) {
            Ok(s) => s,
            Err(_) => return None,
        };
        spiller_side.set_engine_handle(handle_for_spiller);
        let kv_hook: Arc<Mutex<dyn crate::serve::kv_persist::spiller::KvCacheSpill>> =
            Arc::new(Mutex::new(spiller_side));
        Some((kv_hook, bindable))
    }
}

// ---------------------------------------------------------------------------
// Tests.
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::serve::kv_persist::block_store::DiskBlockStore;
    use crate::serve::kv_persist::format::{self, BLOCK_TOKENS};
    use crate::serve::kv_persist::spiller::BlockPrefixCacheSpiller;
    use crate::serve::kv_persist::writer::AsyncWriterHandle;
    use crate::serve::multi_model::{
        KvSpiller, LoadedEngine, LoadedHandle, RestoreOutcome, SpillOutcome,
    };
    use crate::serve::quant_select::QuantType;
    use std::process;
    use std::sync::atomic::{AtomicU32, Ordering};
    use std::time::{Duration, Instant, SystemTime};

    // ---------- harness ----------

    fn temp_dir(label: &str) -> std::path::PathBuf {
        static COUNTER: AtomicU32 = AtomicU32::new(0);
        let n = COUNTER.fetch_add(1, Ordering::SeqCst);
        let pid = process::id();
        let nanos = SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0);
        let dir = std::env::temp_dir()
            .join(format!("hf2q-gemma4-spill-{label}-{pid}-{nanos}-{n}"));
        std::fs::create_dir_all(&dir).expect("temp_dir mkdir");
        dir
    }

    fn try_make_device() -> Option<Arc<MlxDevice>> {
        // CI/test environments without a Metal device return None.
        // The two pure-codec tests (header round-trip / dtype-mismatch
        // CodecErr / short-payload CodecErr) don't need the GPU; they
        // run unconditionally. Tests that DO need the GPU early-return
        // when this returns None.
        MlxDevice::new().ok().map(Arc::new)
    }

    /// Cfg matching a pared-down Gemma 4 shape: 4 layers, sliding
    /// window of 16, head_dim 8, nkv 2 — small enough that 256-token
    /// snapshots are tractable but big enough to exercise multi-head
    /// strides.
    fn small_cfg(dtype: DType) -> Gemma4DenseConfig {
        Gemma4DenseConfig {
            // [Sliding, Sliding, Full, Sliding] — covers both layer
            // types within a single instance.
            layer_types: vec![
                LayerType::Sliding,
                LayerType::Sliding,
                LayerType::Full,
                LayerType::Sliding,
            ],
            nkv_heads: vec![2, 2, 1, 2],
            head_dim: vec![8, 8, 16, 8],
            kv_dtype: dtype,
            sliding_window: 16,
            max_decode_tokens: 32,
        }
    }

    fn fresh_handle_for_engine(device: Arc<MlxDevice>) -> EngineHandle {
        EngineHandle {
            device,
            dense_kvs: Arc::new(RwLock::new(None)),
            write_positions: Arc::new(RwLock::new(Vec::new())),
        }
    }

    /// Build a fully-populated EngineHandle whose dense_kvs is
    /// allocated and filled with deterministic test bytes (one byte
    /// per token slot per head per dim).
    fn populate_handle(
        hook: &Gemma4DenseSpill,
        device: Arc<MlxDevice>,
        seed: u8,
    ) -> EngineHandle {
        let kvs = hook
            .alloc_all_layers(&device, 0)
            .expect("alloc_all_layers");
        // Fill K and V with `seed XOR <layer> XOR <byte_index>` so
        // every byte is distinct and recognisable in test asserts.
        let mut kvs = kvs;
        for (li, layer) in kvs.iter_mut().enumerate() {
            let k_bytes = layer.k.as_mut_slice::<u8>().expect("k as_mut_slice");
            for (i, b) in k_bytes.iter_mut().enumerate() {
                *b = seed ^ (li as u8) ^ ((i as u32) as u8) ^ 0x5A;
            }
            let v_bytes = layer.v.as_mut_slice::<u8>().expect("v as_mut_slice");
            for (i, b) in v_bytes.iter_mut().enumerate() {
                *b = seed ^ (li as u8) ^ ((i as u32) as u8) ^ 0xA5;
            }
        }
        let write_positions = vec![0usize; hook.num_layers()];
        EngineHandle {
            device,
            dense_kvs: Arc::new(RwLock::new(Some(kvs))),
            write_positions: Arc::new(RwLock::new(write_positions)),
        }
    }

    fn fresh_substrate(
        label: &str,
    ) -> (Arc<DiskBlockStore>, Arc<AsyncWriterHandle>, std::path::PathBuf) {
        let dir = temp_dir(label);
        let store = Arc::new(DiskBlockStore::new(dir.clone(), 0).expect("DiskBlockStore"));
        let writer = Arc::new(AsyncWriterHandle::spawn(Arc::clone(&store), 32));
        (store, writer, dir)
    }

    /// Stand-in engine type for the spiller's `LoadedEngine<E>` slot.
    #[derive(Debug)]
    struct TestEngine;

    fn fresh_loaded_engine(
        repo: &str,
        quant: QuantType,
    ) -> Arc<LoadedEngine<TestEngine>> {
        Arc::new(LoadedEngine {
            engine: TestEngine,
            repo: repo.to_string(),
            quant,
            bytes_resident: 1 << 30,
            loaded_at: SystemTime::now(),
        })
    }

    fn fresh_loaded_handle(repo: &str, quant: QuantType) -> LoadedHandle {
        LoadedHandle::new(repo, quant.as_str(), 1 << 30)
    }

    fn wait_for_index_count(store: &Arc<DiskBlockStore>, expected: usize) {
        let deadline = Instant::now() + Duration::from_secs(2);
        while Instant::now() < deadline {
            if store.index().block_count() == expected {
                return;
            }
            std::thread::sleep(Duration::from_millis(5));
        }
        panic!(
            "writer did not drain to expected={expected}; got {} after 2s",
            store.index().block_count()
        );
    }

    // ---------- pure-codec tests (no Metal device required) ----------

    // ===== Test 1: zero-layers config gives zero alignment-neutral hook =====
    #[test]
    fn new_with_zero_layers_returns_zero_alignment_neutral() {
        let cfg = Gemma4DenseConfig {
            layer_types: vec![],
            nkv_heads: vec![],
            head_dim: vec![],
            kv_dtype: DType::F32,
            sliding_window: 16,
            max_decode_tokens: 32,
        };
        let hook = Gemma4DenseSpill::new(cfg).expect("new with zero layers");
        // block_alignment is the format constant, independent of
        // num_layers. A zero-layer hook still returns BLOCK_TOKENS,
        // but every snapshot/restore call returns None / CodecErr
        // because layer_rank >= num_layers.
        assert_eq!(hook.block_alignment(), BLOCK_TOKENS);
        assert_eq!(hook.num_layers(), 0);
        assert!(hook.snapshot_block(0, 0..16).is_none());
    }

    // ===== Test 2: block_alignment returns the format constant =============
    #[test]
    fn block_alignment_returns_256() {
        let hook = Gemma4DenseSpill::new(small_cfg(DType::F32)).expect("new");
        assert_eq!(hook.block_alignment(), BLOCK_TOKENS);
        assert_eq!(BLOCK_TOKENS, 256, "format constant invariant");
    }

    // ===== Test 3: snapshot with no engine handle returns None =============
    #[test]
    fn snapshot_with_no_engine_handle_returns_none() {
        let hook = Gemma4DenseSpill::new(small_cfg(DType::F32)).expect("new");
        // No set_engine_handle call → engine == None.
        let out = hook.snapshot_block(0, 0..16);
        assert!(out.is_none(), "must Skip without an engine handle");
    }

    // ===== Test 4: snapshot with out-of-range layer returns None ===========
    #[test]
    fn snapshot_layer_out_of_range_returns_none() {
        let hook = Gemma4DenseSpill::new(small_cfg(DType::F32)).expect("new");
        // Even with no engine handle the bounds check fires first;
        // verify also with a populated handle.
        assert!(hook.snapshot_block(99, 0..16).is_none());
        let Some(device) = try_make_device() else {
            return;
        };
        let h = populate_handle(&hook, device, 0x42);
        hook.set_engine_handle(h);
        assert!(
            hook.snapshot_block(99, 0..16).is_none(),
            "out-of-range layer skips"
        );
    }

    // ===== Test 5: snapshot full layer round-trips dtype + bytes ===========
    #[test]
    fn snapshot_full_layer_returns_dtype_and_bytes() {
        let Some(device) = try_make_device() else {
            return;
        };
        let hook =
            Gemma4DenseSpill::new(small_cfg(DType::F32)).expect("new");
        let handle = populate_handle(&hook, device.clone(), 0x42);
        hook.set_engine_handle(handle);
        // Layer 2 is full-attention, head_dim 16, nkv 1. capacity =
        // max_decode_tokens (seq_len_hint=0) = 32.
        let layer = 2usize;
        let range = 0u32..8u32;
        let payload = hook
            .snapshot_block(layer, range.clone())
            .expect("snapshot");
        // Header parses cleanly.
        let hdr = PayloadHeader::from_bytes(&payload).expect("hdr parse");
        assert_eq!(hdr.dtype, DtypeTag::F32);
        assert!(!hdr.is_sliding);
        assert_eq!(hdr.nkv_heads, 1);
        assert_eq!(hdr.head_dim, 16);
        assert_eq!(hdr.range_start, 0);
        assert_eq!(hdr.range_end, 8);
        // n_tokens * nkv * head_dim * 4 (F32) = 8 * 1 * 16 * 4 = 512.
        assert_eq!(hdr.k_byte_len, 512);
        assert_eq!(hdr.v_byte_len, 512);
        // Linear layer write_pos is the sentinel.
        assert_eq!(hdr.write_pos, WRITE_POS_LINEAR_SENTINEL);
        // Total size: header + 512 + 512.
        assert_eq!(payload.len(), PAYLOAD_HEADER_BYTES + 1024);
    }

    // ===== Test 6: snapshot sliding layer handles ring wrap ================
    // (Critical edge case: range straddles capacity boundary.)
    #[test]
    fn snapshot_sliding_layer_handles_ring_wrap() {
        let Some(device) = try_make_device() else {
            return;
        };
        let hook =
            Gemma4DenseSpill::new(small_cfg(DType::F32)).expect("new");
        // Layer 0 is sliding, sliding_window = 16, nkv = 2, head_dim = 8.
        let handle = populate_handle(&hook, device.clone(), 0x10);
        hook.set_engine_handle(handle);
        let layer = 0usize;
        // Range [12..20) straddles the ring boundary at 16.
        // Slot map: tok 12→slot 12, tok 13→13, ..., tok 15→15,
        //           tok 16→slot 0, tok 17→1, tok 18→2, tok 19→3.
        let range = 12u32..20u32;
        let payload = hook
            .snapshot_block(layer, range.clone())
            .expect("snapshot wrap");
        let hdr = PayloadHeader::from_bytes(&payload).expect("hdr parse");
        assert!(hdr.is_sliding, "sliding flag set");
        assert_eq!(hdr.capacity, 16);
        assert_eq!(hdr.range_start, 12);
        assert_eq!(hdr.range_end, 20);
        // 8 tokens * 2 nkv * 8 head_dim * 4 (F32) = 512 bytes per K/V.
        assert_eq!(hdr.k_byte_len, 512);
        assert_eq!(hdr.v_byte_len, 512);

        // Manually reconstruct what the snapshot bytes SHOULD contain:
        // for each head h in 0..2, for each tok in 12..20, we expect
        // the bytes that populate_handle wrote at slot = tok % 16.
        // populate_handle's K seed pattern is
        // `seed XOR layer XOR (i as u32 as u8) XOR 0x5A`
        // where i is the linear byte offset within the layer's K
        // buffer. Layer's K shape is [2, 16, 8] F32 → 2*16*8*4 = 1024
        // bytes. For head 0 the byte index range is [0..512), for
        // head 1 [512..1024).
        let seed: u8 = 0x10;
        let layer_idx: u8 = 0;
        let mut expected_k = Vec::with_capacity(512);
        for h in 0..2usize {
            for tok in 12u32..20 {
                let slot = (tok as usize) % 16;
                // 8 elements * 4 bytes = 32 bytes per token chunk.
                let head_base = h * 16 * 8 * 4;
                let off = head_base + slot * 8 * 4;
                for k in 0..32 {
                    let i = (off + k) as u32 as u8;
                    expected_k.push(seed ^ layer_idx ^ i ^ 0x5A);
                }
            }
        }
        let body_start = PAYLOAD_HEADER_BYTES;
        let k_slice = &payload[body_start..body_start + 512];
        assert_eq!(
            k_slice,
            &expected_k[..],
            "ring-wrap K bytes equal token-order slot reads"
        );
    }

    // ===== Test 7: F16 dtype round-trips through the header ================
    #[test]
    fn snapshot_f16_dtype_payload_round_trips() {
        let Some(device) = try_make_device() else {
            return;
        };
        let hook =
            Gemma4DenseSpill::new(small_cfg(DType::F16)).expect("new f16");
        let handle = populate_handle(&hook, device.clone(), 0x77);
        hook.set_engine_handle(handle);
        let layer = 0usize;
        let payload = hook
            .snapshot_block(layer, 0..8)
            .expect("snapshot f16");
        let hdr = PayloadHeader::from_bytes(&payload).expect("hdr parse");
        assert_eq!(hdr.dtype, DtypeTag::F16);
        // 8 tokens * 2 nkv * 8 head_dim * 2 (F16) = 256 bytes per K/V.
        assert_eq!(hdr.k_byte_len, 256);
        assert_eq!(hdr.v_byte_len, 256);
    }

    // ===== Test 8: restore allocates dense_kvs first =======================
    #[test]
    fn restore_with_dense_kvs_none_allocates_first() {
        let Some(device) = try_make_device() else {
            return;
        };
        let mut hook =
            Gemma4DenseSpill::new(small_cfg(DType::F32)).expect("new");
        // Step 1: snapshot from a populated handle to get a real
        // payload.
        let producer_handle = populate_handle(&hook, device.clone(), 0xAB);
        hook.set_engine_handle(producer_handle);
        let payload = hook
            .snapshot_block(0, 0..8)
            .expect("snapshot for restore");
        // Step 2: install a NEW handle whose dense_kvs is None — the
        // post_admit-before-prefill case.
        let consumer_handle = fresh_handle_for_engine(device.clone());
        hook.set_engine_handle(consumer_handle.clone());
        assert!(
            consumer_handle.dense_kvs.read().unwrap().is_none(),
            "precondition: dense_kvs starts None"
        );
        // Step 3: restore.
        hook.restore_block(0, 0..8, &payload)
            .expect("restore alloc-then-write");
        // Post: dense_kvs is Some(Vec) of length num_layers.
        let kvs = consumer_handle.dense_kvs.read().unwrap();
        let kvs = kvs.as_ref().expect("dense_kvs allocated");
        assert_eq!(kvs.len(), hook.num_layers());
    }

    // ===== Test 9: restore with dtype mismatch returns CodecErr ============
    #[test]
    fn restore_dtype_mismatch_returns_codec_err() {
        let Some(device) = try_make_device() else {
            return;
        };
        // Producer is F32; consumer hook expects F16. Restore must
        // reject — never mutate the cache.
        let producer = Gemma4DenseSpill::new(small_cfg(DType::F32)).expect("new f32");
        let producer_handle = populate_handle(&producer, device.clone(), 0xCC);
        producer.set_engine_handle(producer_handle);
        let payload = producer
            .snapshot_block(0, 0..4)
            .expect("snapshot f32");
        let mut consumer =
            Gemma4DenseSpill::new(small_cfg(DType::F16)).expect("new f16");
        consumer.set_engine_handle(fresh_handle_for_engine(device));
        let err = consumer.restore_block(0, 0..4, &payload).unwrap_err();
        assert_eq!(err, SpillErrorKind::CodecErr, "dtype mismatch → CodecErr");
    }

    // ===== Test 10: restore with layer-type mismatch returns CodecErr ======
    #[test]
    fn restore_layer_type_mismatch_returns_codec_err() {
        let Some(device) = try_make_device() else {
            return;
        };
        let producer = Gemma4DenseSpill::new(small_cfg(DType::F32)).expect("p");
        let h = populate_handle(&producer, device.clone(), 0x33);
        producer.set_engine_handle(h);
        // Snapshot from layer 0 (sliding).
        let payload = producer
            .snapshot_block(0, 0..4)
            .expect("snapshot sliding");
        // Consumer expects layer 0 to be Full — flip the layer_types.
        let mut bad_cfg = small_cfg(DType::F32);
        bad_cfg.layer_types[0] = LayerType::Full;
        let mut consumer = Gemma4DenseSpill::new(bad_cfg).expect("c");
        consumer.set_engine_handle(fresh_handle_for_engine(device));
        let err = consumer.restore_block(0, 0..4, &payload).unwrap_err();
        assert_eq!(err, SpillErrorKind::CodecErr);
    }

    // ===== Test 11: R-C1 — full pre_evict→post_admit round-trip byte exact =
    // (Load-bearing for B-dense.2.)
    #[test]
    fn pre_evict_then_post_admit_round_trip_byte_exact() {
        let Some(device) = try_make_device() else {
            return;
        };
        let (store, writer, dir) = fresh_substrate("rc1");
        let spiller: BlockPrefixCacheSpiller<TestEngine> =
            BlockPrefixCacheSpiller::new(Arc::clone(&store), Arc::clone(&writer));

        // Hook: 1-layer full-attention shape so we exercise the linear
        // path end-to-end. nkv=1, head_dim=8, capacity=64 (linear
        // = max_decode_tokens; sliding_window unused).
        let cfg = Gemma4DenseConfig {
            layer_types: vec![LayerType::Full],
            nkv_heads: vec![1],
            head_dim: vec![8],
            kv_dtype: DType::F32,
            sliding_window: 16,
            max_decode_tokens: 256,
        };
        let hook = Gemma4DenseSpill::new(cfg).expect("new");
        let producer_handle = populate_handle(&hook, device.clone(), 0x51);
        hook.set_engine_handle(producer_handle.clone());

        let arc_hook: Arc<Mutex<dyn KvCacheSpill>> = Arc::new(Mutex::new(hook));
        spiller.register_family("acme/g4-rc1".into(), QuantType::Q4_K_M, arc_hook.clone());

        // Spill phase.
        let handle = fresh_loaded_handle("acme/g4-rc1", QuantType::Q4_K_M);
        let engine = fresh_loaded_engine("acme/g4-rc1", QuantType::Q4_K_M);
        let spill = spiller.pre_evict(&handle, &engine);
        match spill {
            SpillOutcome::EnqueuedBlocks(n) if n >= 1 => {}
            other => panic!("expected EnqueuedBlocks(>=1); got {other:?}"),
        }

        wait_for_index_count(&store, 1);

        // Read the on-disk envelope back via the canonical reader so
        // the block_hash + sha verify path runs.
        let metas = store.index().snapshot_all();
        assert_eq!(metas.len(), 1);
        let (header, body) =
            format::read_envelope_body(&metas[0].file_path).expect("envelope round-trips");
        assert_eq!(header.format_version, format::CURRENT_FORMAT_VERSION.0);

        // Body bytes are the exact payload our snapshot_block produced.
        // Decode the payload header and verify a chunk of K against
        // what populate_handle wrote.
        let phdr = PayloadHeader::from_bytes(&body).expect("phdr");
        assert_eq!(phdr.dtype, DtypeTag::F32);
        assert!(!phdr.is_sliding);
        assert_eq!(phdr.range_start, 0);
        // The spiller's A.3 ranges_for_layer ships [0..BLOCK_TOKENS),
        // so for our 1-layer-full-attn hook we get 256 tokens. n =
        // 256 * 1 * 8 * 4 = 8192 bytes per K/V.
        assert_eq!(phdr.range_end, BLOCK_TOKENS);
        assert_eq!(phdr.k_byte_len, 8192);
        assert_eq!(phdr.v_byte_len, 8192);

        // Build a consumer hook with dense_kvs = None — the post_admit
        // case the production engine actually hits.
        let cfg2 = Gemma4DenseConfig {
            layer_types: vec![LayerType::Full],
            nkv_heads: vec![1],
            head_dim: vec![8],
            kv_dtype: DType::F32,
            sliding_window: 16,
            max_decode_tokens: 256,
        };
        let consumer = Gemma4DenseSpill::new(cfg2).expect("consumer new");
        let consumer_handle = fresh_handle_for_engine(device.clone());
        consumer.set_engine_handle(consumer_handle.clone());
        let arc_consumer: Arc<Mutex<dyn KvCacheSpill>> =
            Arc::new(Mutex::new(consumer));
        // Re-register under the same key — the spiller's
        // register_family overwrites.
        spiller.register_family(
            "acme/g4-rc1".into(),
            QuantType::Q4_K_M,
            arc_consumer.clone(),
        );

        // Restore phase.
        let restore = spiller.post_admit("acme/g4-rc1", QuantType::Q4_K_M, &engine);
        match restore {
            RestoreOutcome::RestoredBlocks(n) if n >= 1 => {}
            other => panic!("expected RestoredBlocks(>=1); got {other:?}"),
        }

        // Verify dense_kvs is now allocated.
        let kvs = consumer_handle.dense_kvs.read().unwrap();
        let kvs = kvs.as_ref().expect("allocated");
        assert_eq!(kvs.len(), 1);

        // Compare every byte in layer 0's K against what the producer
        // wrote at slot[tok] for tok in 0..256, head 0 (nkv=1).
        // populate_handle's K seed pattern: seed XOR layer XOR
        // (i as u32 as u8) XOR 0x5A.
        let seed: u8 = 0x51;
        let layer_idx: u8 = 0;
        let consumer_k = kvs[0].k.as_slice::<u8>().expect("k as_slice");
        let consumer_v = kvs[0].v.as_slice::<u8>().expect("v as_slice");
        // Linear layer: byte i is at offset i (head 0 only since nkv=1).
        // Capacity is 256 (max_decode_tokens) so tokens 0..256 occupy
        // bytes 0..(256*8*4) = 0..8192.
        for i in 0..8192usize {
            let expected_k = seed ^ layer_idx ^ (i as u32 as u8) ^ 0x5A;
            let expected_v = seed ^ layer_idx ^ (i as u32 as u8) ^ 0xA5;
            assert_eq!(
                consumer_k[i], expected_k,
                "K byte {i}: got {} expected {}",
                consumer_k[i], expected_k
            );
            assert_eq!(
                consumer_v[i], expected_v,
                "V byte {i}: got {} expected {}",
                consumer_v[i], expected_v
            );
        }
        eprintln!(
            "[R-C1] PASS — Gemma4DenseSpill round-trip 8192 K + 8192 V bytes byte-exact via spill→disk→restore"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    // ===== Test 12: ring-wrap round-trip byte-exact ========================
    // (Spec calls this out as load-bearing for sliding layers.)
    #[test]
    fn pre_evict_then_post_admit_round_trip_byte_exact_sliding_with_wrap() {
        let Some(device) = try_make_device() else {
            return;
        };
        // Producer: 1 sliding layer with sliding_window = 8, nkv = 1,
        // head_dim = 4. Range [4..12) wraps (slots [4,5,6,7,0,1,2,3]).
        let cfg = Gemma4DenseConfig {
            layer_types: vec![LayerType::Sliding],
            nkv_heads: vec![1],
            head_dim: vec![4],
            kv_dtype: DType::F32,
            sliding_window: 8,
            max_decode_tokens: 16,
        };
        let producer = Gemma4DenseSpill::new(cfg.clone()).expect("p new");
        let producer_handle = populate_handle(&producer, device.clone(), 0x99);
        // Set a non-zero write_pos to verify it's preserved through
        // the round trip.
        {
            let mut wp = producer_handle.write_positions.write().unwrap();
            wp[0] = 5;
        }
        producer.set_engine_handle(producer_handle.clone());

        // Snapshot the wrapping range directly (no spiller in this
        // test — we exercise the codec end-to-end without the
        // envelope wrapper to keep the assertion focused on the
        // ring-wrap edge case).
        let range = 4u32..12u32;
        let payload = producer
            .snapshot_block(0, range.clone())
            .expect("snapshot wrap");

        // Verify the payload header records the wrap range.
        let phdr = PayloadHeader::from_bytes(&payload).expect("phdr");
        assert!(phdr.is_sliding, "sliding flag");
        assert_eq!(phdr.capacity, 8);
        assert_eq!(phdr.range_start, 4);
        assert_eq!(phdr.range_end, 12);
        assert_eq!(phdr.write_pos, 5, "write_pos preserved");
        // 8 tokens * 1 nkv * 4 head_dim * 4 (F32) = 128 bytes per K/V.
        assert_eq!(phdr.k_byte_len, 128);

        // Build a consumer with dense_kvs = None.
        let mut consumer = Gemma4DenseSpill::new(cfg).expect("c new");
        let consumer_handle = fresh_handle_for_engine(device);
        consumer.set_engine_handle(consumer_handle.clone());

        // Restore.
        consumer
            .restore_block(0, range.clone(), &payload)
            .expect("restore wrap");

        // Verify byte-exact: for each token in [4..12), the consumer's
        // K at slot (tok % 8) holds the same 4*4=16 bytes the producer
        // had.
        let producer_kvs = producer_handle.dense_kvs.read().unwrap();
        let producer_k = producer_kvs.as_ref().unwrap()[0]
            .k
            .as_slice::<u8>()
            .expect("p k");
        let consumer_kvs = consumer_handle.dense_kvs.read().unwrap();
        let consumer_k = consumer_kvs.as_ref().unwrap()[0]
            .k
            .as_slice::<u8>()
            .expect("c k");

        let head_dim_bytes = 4 * 4; // head_dim * F32 elem_bytes
        for tok in 4u32..12 {
            let slot = (tok as usize) % 8;
            let off = slot * head_dim_bytes;
            // Compare 16 bytes per token.
            for k in 0..head_dim_bytes {
                assert_eq!(
                    consumer_k[off + k],
                    producer_k[off + k],
                    "ring-wrap byte mismatch at tok={tok} slot={slot} k={k}"
                );
            }
        }

        // Verify write_pos restored.
        let wp = consumer_handle.write_positions.read().unwrap();
        assert_eq!(wp.len(), 1);
        assert_eq!(wp[0], 5, "write_pos restored");
        eprintln!(
            "[ring-wrap] PASS — sliding layer range [4..12) over capacity=8 round-tripped byte-exact"
        );
    }

    // ===== Test 13: short payload returns CodecErr =========================
    #[test]
    fn restore_with_short_payload_returns_codec_err() {
        let Some(device) = try_make_device() else {
            return;
        };
        let mut hook =
            Gemma4DenseSpill::new(small_cfg(DType::F32)).expect("new");
        hook.set_engine_handle(fresh_handle_for_engine(device));
        // Header size is 34 bytes; pass only 10 → must be rejected
        // before any further work.
        let bad = vec![0u8; 10];
        let err = hook.restore_block(0, 0..4, &bad).unwrap_err();
        assert_eq!(err, SpillErrorKind::CodecErr);
        // Header-only (no body) is also short.
        let mut hdr_only = vec![0u8; PAYLOAD_HEADER_BYTES];
        hdr_only[0..4].copy_from_slice(PAYLOAD_MAGIC);
        // The header parse will reject because is_sliding byte (5) is
        // 0 (full-attention) but layer 0 in small_cfg is sliding —
        // CodecErr surfaces from validate_header. That's still the
        // expected outcome.
        let err2 = hook.restore_block(0, 0..4, &hdr_only).unwrap_err();
        assert_eq!(err2, SpillErrorKind::CodecErr);
    }

    // ===== Test 14: long payload returns CodecErr ==========================
    #[test]
    fn restore_with_long_payload_returns_codec_err() {
        let Some(device) = try_make_device() else {
            return;
        };
        // Build a real payload, then append junk bytes to make it
        // longer than `header + k_byte_len + v_byte_len`.
        let producer = Gemma4DenseSpill::new(small_cfg(DType::F32)).expect("p");
        producer.set_engine_handle(populate_handle(&producer, device.clone(), 0x44));
        let mut payload = producer
            .snapshot_block(0, 0..4)
            .expect("snapshot");
        let original_len = payload.len();
        payload.extend_from_slice(&[0xDEu8, 0xAD, 0xBE, 0xEF]);
        assert_eq!(payload.len(), original_len + 4);

        let mut consumer = Gemma4DenseSpill::new(small_cfg(DType::F32)).expect("c");
        consumer.set_engine_handle(fresh_handle_for_engine(device));
        let err = consumer.restore_block(0, 0..4, &payload).unwrap_err();
        assert_eq!(err, SpillErrorKind::CodecErr);
    }

    // ===== Test 15: engine handle set then clear disables snapshot =========
    #[test]
    fn engine_handle_set_then_clear_disables_snapshot() {
        let Some(device) = try_make_device() else {
            return;
        };
        let hook = Gemma4DenseSpill::new(small_cfg(DType::F32)).expect("new");
        // 1) No handle: snapshot returns None.
        assert!(hook.snapshot_block(0, 0..4).is_none());
        // 2) Set: snapshot returns Some.
        hook.set_engine_handle(populate_handle(&hook, device, 0x77));
        assert!(hook.snapshot_block(0, 0..4).is_some());
        // 3) Clear: snapshot returns None again.
        hook.clear_engine_handle();
        assert!(hook.snapshot_block(0, 0..4).is_none());
    }

    // ===== Test 16 (extra): multi-layer round-trip =========================
    #[test]
    fn multiple_layers_round_trip_byte_exact() {
        let Some(device) = try_make_device() else {
            return;
        };
        let producer = Gemma4DenseSpill::new(small_cfg(DType::F32)).expect("p");
        producer.set_engine_handle(populate_handle(&producer, device.clone(), 0x21));

        let mut consumer =
            Gemma4DenseSpill::new(small_cfg(DType::F32)).expect("c");
        consumer.set_engine_handle(fresh_handle_for_engine(device));

        // For each of the 4 layers, snapshot a 4-token range and
        // restore it. After the round-trip the consumer's K/V
        // bytes for the touched range must equal the producer's.
        for li in 0..producer.num_layers() {
            let nkv = producer.nkv_heads[li];
            let hd = producer.head_dim[li];
            let dtype_bytes = producer.kv_dtype.size_of();
            let n_tokens = 4u32;
            let range = 0u32..n_tokens;
            let payload = producer
                .snapshot_block(li, range.clone())
                .unwrap_or_else(|| panic!("snapshot L{li}"));
            consumer
                .restore_block(li, range.clone(), &payload)
                .unwrap_or_else(|e| panic!("restore L{li}: {e:?}"));

            // Verify byte-exact for the touched range.
            let producer_kvs = {
                let g = producer
                    .engine
                    .read()
                    .unwrap();
                let h = g.as_ref().unwrap();
                let kvs = h.dense_kvs.read().unwrap();
                Arc::clone(&h.dense_kvs)
                    .read()
                    .unwrap()
                    .as_ref()
                    .map(|_| ())
                    .unwrap();
                drop(kvs);
                Arc::clone(&h.dense_kvs)
            };
            let consumer_kvs = {
                let g = consumer.engine.read().unwrap();
                let h = g.as_ref().unwrap();
                Arc::clone(&h.dense_kvs)
            };
            let p_g = producer_kvs.read().unwrap();
            let c_g = consumer_kvs.read().unwrap();
            let p_layer = &p_g.as_ref().unwrap()[li];
            let c_layer = &c_g.as_ref().unwrap()[li];
            let p_k = p_layer.k.as_slice::<u8>().expect("p k");
            let c_k = c_layer.k.as_slice::<u8>().expect("c k");
            // Layer 2 is full-attn, others sliding. For full-attn
            // tokens 0..4 occupy slots 0..4. For sliding (capacity
            // 16) the same. We compare those slot ranges.
            let token_chunk = nkv * hd * dtype_bytes;
            let cap = match producer.layer_types[li] {
                LayerType::Sliding => producer.sliding_window,
                LayerType::Full => producer.max_decode_tokens,
            };
            let head_stride = cap * hd * dtype_bytes;
            for h in 0..nkv {
                let head_base = h * head_stride;
                for tok in 0..n_tokens {
                    let slot = (tok as usize) % cap;
                    let off = head_base + slot * hd * dtype_bytes;
                    assert_eq!(
                        &c_k[off..off + hd * dtype_bytes],
                        &p_k[off..off + hd * dtype_bytes],
                        "L{li} h{h} tok{tok}"
                    );
                }
            }
            let _ = token_chunk;
        }
    }

    // ===== Test 17 (extra): concurrent snapshot calls serialize via Mutex =
    #[test]
    fn concurrent_snapshot_calls_serialize_via_mutex() {
        let Some(device) = try_make_device() else {
            return;
        };
        let hook = Gemma4DenseSpill::new(small_cfg(DType::F32)).expect("new");
        hook.set_engine_handle(populate_handle(&hook, device, 0x88));
        // Wrap in Arc<Mutex<dyn KvCacheSpill>> to mirror how the
        // spiller holds the hook.
        let arc_hook: Arc<Mutex<dyn KvCacheSpill>> = Arc::new(Mutex::new(hook));
        let n = 8usize;
        let mut joins = Vec::with_capacity(n);
        for _ in 0..n {
            let h = Arc::clone(&arc_hook);
            joins.push(std::thread::spawn(move || {
                let g = h.lock().expect("lock");
                let out = g.snapshot_block(0, 0..4);
                assert!(out.is_some());
                out.unwrap().len()
            }));
        }
        let mut sizes = Vec::with_capacity(n);
        for j in joins {
            sizes.push(j.join().expect("join"));
        }
        // All threads return the same payload size — proves the lock
        // serialized them and the underlying state was stable.
        let first = sizes[0];
        for s in &sizes {
            assert_eq!(*s, first);
        }
    }

    // ===== Phase C.1 EngineBindable tests ==================================

    /// Spec test 9: round-trip — bind a handle via the EngineBindable
    /// trait, snapshot succeeds; unbind via the trait, snapshot
    /// returns None.
    #[test]
    fn engine_bindable_gemma4_dense_round_trip_set_then_clear() {
        use crate::serve::kv_persist::EngineBindable;
        let Some(device) = try_make_device() else {
            return;
        };
        let hook = Gemma4DenseSpill::new(small_cfg(DType::F32)).expect("new");

        // Build a populated EngineHandle and wrap in Arc.
        let handle = populate_handle(&hook, device.clone(), 0xC1);
        let handle_arc: Arc<EngineHandle> = Arc::new(handle);
        let dyn_view: Arc<dyn std::any::Any + Send + Sync> = handle_arc.clone();

        // Pre-bind: no engine handle ⇒ snapshot returns None.
        assert!(hook.snapshot_block(0, 0..4).is_none());

        // Bind via EngineBindable trait.
        EngineBindable::bind_engine(&hook, dyn_view);
        // Post-bind: snapshot now returns Some.
        let out = hook.snapshot_block(0, 0..4);
        assert!(out.is_some(), "snapshot active after bind_engine");
        assert!(out.unwrap().len() > PAYLOAD_HEADER_BYTES);

        // Unbind via EngineBindable trait.
        EngineBindable::unbind_engine(&hook);
        // Post-unbind: snapshot returns None again.
        assert!(
            hook.snapshot_block(0, 0..4).is_none(),
            "snapshot disabled after unbind_engine"
        );

        // The original Arc<EngineHandle> is still live; the bind
        // cloned the inner handle but did NOT retain the outer Arc.
        assert_eq!(
            Arc::strong_count(&handle_arc),
            1,
            "EngineBindable did not retain the outer Arc"
        );
    }

    /// Spec test 10: downcast to wrong concrete type silently no-ops.
    /// The hook's bind_engine never panics on type mismatch.
    #[test]
    fn engine_bindable_gemma4_dense_downcast_wrong_type_returns_silently() {
        use crate::serve::kv_persist::EngineBindable;
        let hook = Gemma4DenseSpill::new(small_cfg(DType::F32)).expect("new");

        // Pre-bind: no handle.
        assert!(
            hook.snapshot_block(0, 0..4).is_none(),
            "no handle pre-bind"
        );

        // Pass a String wrapped in Arc<dyn Any>. Downcast to
        // EngineHandle MUST fail; the impl drops it silently.
        let bogus: Arc<dyn std::any::Any + Send + Sync> =
            Arc::new(String::from("not-an-engine-handle"));
        // This must NOT panic.
        EngineBindable::bind_engine(&hook, bogus);

        // Post-bind: still no handle (downcast failed silently).
        assert!(
            hook.snapshot_block(0, 0..4).is_none(),
            "snapshot stays disabled after type-mismatch bind"
        );

        // unbind_engine on an unbound hook is also a no-op (does
        // not panic).
        EngineBindable::unbind_engine(&hook);
    }

    /// Spec test 11 (additional): the C.1 LoaderWrapper builds an
    /// `Arc<E>` where E is the loaded Engine and passes that to
    /// `bind_for`. For Gemma4DenseSpill, E is NOT EngineHandle, so the
    /// downcast fails silently and the engine slot stays None. The
    /// production cmd_serve path then drives a SEPARATE bind with an
    /// `Arc<EngineHandle>` constructed from the loaded engine, which
    /// succeeds.
    #[test]
    fn engine_bindable_gemma4_dense_silently_ignores_non_handle_arc() {
        use crate::serve::kv_persist::EngineBindable;
        let hook = Gemma4DenseSpill::new(small_cfg(DType::F32)).expect("new");

        // Synthetic "engine" Arc — not an EngineHandle.
        struct FakeEngine {
            _marker: u32,
        }
        let fake = Arc::new(FakeEngine { _marker: 7 });
        let dyn_view: Arc<dyn std::any::Any + Send + Sync> = fake;

        EngineBindable::bind_engine(&hook, dyn_view);

        // No engine bound ⇒ snapshot is None.
        assert!(hook.snapshot_block(0, 0..4).is_none());
    }

    // ===== Phase B-dense.2 — FamilyHookFactory tests ========================

    /// Spec test 5: `Gemma4DenseSpill::try_from_engine_arc` succeeds
    /// on `Arc<EngineHandle>` and yields a spill with engine bound.
    /// Falsifier: returns None on type match, or doesn't bind the
    /// engine (hook.snapshot_block returns None despite a populated
    /// dense_kvs).
    #[test]
    fn factory_try_from_engine_arc_succeeds_on_handle_match() {
        use crate::serve::kv_persist::EngineBindable;

        let device = match try_make_device() {
            Some(d) => d,
            None => {
                eprintln!(
                    "[B-dense.2 factory] no MlxDevice — skipping populated-handle test"
                );
                return;
            }
        };

        // Build a populated handle outside the factory so the test
        // verifies "engine wired" by observing snapshot_block returns
        // Some(_).
        let scratch_hook = Gemma4DenseSpill::new(small_cfg(DType::F32)).expect("new");
        let handle = populate_handle(&scratch_hook, device, 0x55);

        let handle_arc: Arc<EngineHandle> = Arc::new(handle);
        let dyn_view: Arc<dyn std::any::Any + Send + Sync> = handle_arc;

        let spill = Gemma4DenseSpill::try_from_engine_arc(small_cfg(DType::F32), dyn_view)
            .expect("try_from_engine_arc on matching Arc<EngineHandle> ⇒ Some(spill)");

        // The spill is wired: snapshot_block returns Some on a
        // populated layer + range. (We can't observe the bind via
        // EngineBindable because try_from_engine_arc bypasses that
        // path; the snapshot path is the load-bearing observation.)
        let bytes = spill.snapshot_block(0, 0..4);
        assert!(
            bytes.is_some(),
            "factory-built spill yields snapshot bytes on populated layer"
        );

        // The unbind path also works (no panic, engine slot clears).
        EngineBindable::unbind_engine(&spill);
        assert!(spill.snapshot_block(0, 0..4).is_none());
    }

    /// Spec test 6: `Gemma4DenseSpill::try_from_engine_arc` returns
    /// None on type mismatch — without panic. Mirrors the
    /// `engine_bindable_gemma4_dense_silently_ignores_non_handle_arc`
    /// test's payload.
    #[test]
    fn factory_try_from_engine_arc_returns_none_on_type_mismatch() {
        struct FakeNonHandle {
            _marker: u32,
        }
        let fake = Arc::new(FakeNonHandle { _marker: 9 });
        let dyn_view: Arc<dyn std::any::Any + Send + Sync> = fake;

        let result = Gemma4DenseSpill::try_from_engine_arc(small_cfg(DType::F32), dyn_view);
        assert!(result.is_none(), "type mismatch ⇒ None, no panic");
    }

    /// Spec test 7: `Gemma4DenseSpillFactory::try_construct` round-
    /// trips through the `FamilyHookFactory` trait with a matching
    /// engine.
    ///
    /// Falsifier: returns None on type match, or the bindable_hook
    /// half is not actually wired (subsequent unbind_engine fails to
    /// clear the engine slot).
    #[test]
    fn factory_construct_from_matching_engine_returns_some_tuple() {
        use crate::serve::kv_persist::registry::FamilyHookFactory;
        use crate::serve::kv_persist::EngineBindable;

        let device = match try_make_device() {
            Some(d) => d,
            None => {
                eprintln!(
                    "[B-dense.2 factory] no MlxDevice — skipping factory-construct test"
                );
                return;
            }
        };

        // Populated handle (mirrors what cmd_serve will assemble from
        // the live MlxModelWeights post-load).
        let scratch_hook = Gemma4DenseSpill::new(small_cfg(DType::F32)).expect("new");
        let handle = populate_handle(&scratch_hook, device, 0x66);
        let handle_arc: Arc<EngineHandle> = Arc::new(handle);
        let dyn_view: Arc<dyn std::any::Any + Send + Sync> = handle_arc;

        let factory = Gemma4DenseSpillFactory::new(small_cfg(DType::F32));
        let result = factory.try_construct(dyn_view);

        let (kv_hook, bindable_hook) = result.expect("matching engine ⇒ Some(tuple)");

        // The kv side has block_alignment = BLOCK_TOKENS (the
        // Gemma4DenseSpill impl returns the format constant).
        let alignment = kv_hook
            .lock()
            .expect("kv_hook Mutex poisoned in test")
            .block_alignment();
        assert_eq!(alignment, BLOCK_TOKENS, "kv_hook is a real Gemma4DenseSpill");

        // The bindable side is wired — unbind clears its engine slot
        // (we can't directly observe, but no-panic + idempotent
        // re-call is the contract).
        EngineBindable::unbind_engine(bindable_hook.as_ref());
        EngineBindable::unbind_engine(bindable_hook.as_ref());
    }

    /// Spec test 8: `Gemma4DenseSpillFactory::try_construct` returns
    /// None when the engine_dyn doesn't downcast to
    /// `Arc<EngineHandle>`. Mirrors the auto-LoaderWrapper-bind path
    /// where `Arc<Engine>` is delivered.
    #[test]
    fn factory_construct_from_wrong_engine_type_returns_none() {
        use crate::serve::kv_persist::registry::FamilyHookFactory;

        struct FakeEngine {
            _marker: u32,
        }
        let fake = Arc::new(FakeEngine { _marker: 11 });
        let dyn_view: Arc<dyn std::any::Any + Send + Sync> = fake;

        let factory = Gemma4DenseSpillFactory::new(small_cfg(DType::F32));
        let result = factory.try_construct(dyn_view);
        assert!(result.is_none(), "non-EngineHandle ⇒ None");
    }
}
