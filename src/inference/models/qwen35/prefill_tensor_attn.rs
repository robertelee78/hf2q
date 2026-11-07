//! ADR-063 gap program F3: tensor-API flash attention for resumed Qwen
//! prefill on TQ-only full-attention slots.
//!
//! Route (c′) replaces the BF16 bridge (route (c),
//! [`super::gpu_full_attn::apply_flash_attn_prefill_seq_major_resume_via_tq_cache_for_slot`])
//! for chunks with a cached prefix (`cur_len > 0`) and more than 32 queries,
//! when `HF2Q_QWEN_PREFILL_TENSOR_ATTN` is on and the device has the Metal
//! tensor API. Per layer, one command buffer without a host wait:
//! 1. TQ K and V over `[0, L)` → F16 staging in the FWHT-rotated domain
//!    (`tq_dequantize_hb_kv_seq_f16`), and Q (`q_rope`) rotated in place
//!    (`fwht_sign_premult_f32`). The rotation is orthonormal, so attention in
//!    the rotated domain equals attention in the original one.
//! 2. First FA layer of the chunk only: the causal F16 mask
//!    (`q_abs_offset = cur_len`) and its blk classification. Both depend only
//!    on `(cur_len, seq_len)` and are shared by every FA layer of the chunk.
//! 3. One `flash_attn_prefill_tensor_f16_d256` dispatch (scale 1/16 folded
//!    into Q) writing F32 `[seq_len, n_heads, head_dim]`.
//! 4. `fwht_sign_undo_f32` on the output, then `commit_labeled`.
//!
//! Numerics: Q1 Class B against route (c) (F16 inputs and probabilities
//! against BF16), never bitwise. Rows are invariant to how a suffix is split
//! into chunks of more than 32 queries, and to the query-block split below,
//! because key tiles start at absolute position 0.
//!
//! The per-chunk staging, mask and blk live in a thread-local
//! [`TensorResumeScope`] opened by `forward_gpu_impl` around the layer loop;
//! outside a scope (MTP drafts, direct helpers) resumed prefill keeps the
//! bridge.

use std::cell::RefCell;

use anyhow::{anyhow, ensure, Context, Result};
use mlx_native::ops::flash_attn_prefill_blk::{
    alloc_blk_buffer, dispatch_flash_attn_prefill_blk_f16, BlkParams,
};
use mlx_native::ops::flash_attn_prefill_mask::{build_sdpa_mask_f16, SdpaMaskParams};
use mlx_native::ops::flash_attn_prefill_tensor as tensor;
use mlx_native::ops::fwht_standalone::{
    dispatch_fwht_sign_premult_f32, dispatch_fwht_sign_undo_f32,
};
use mlx_native::{CommandEncoder, DType, KernelRegistry, MlxBuffer, MlxDevice};

use super::kv_cache::FullAttnKvSlot;
use crate::serve::multi_seq_kv::SlotId;

/// Opt-in switch for route (c′): `1`/`on`/`true` requests the tensor route,
/// unset or `0`/`off`/`false` keeps the bridge.
pub(crate) const QWEN_PREFILL_TENSOR_ATTN_ENV: &str = "HF2Q_QWEN_PREFILL_TENSOR_ATTN";

/// Largest F16 mask one dispatch may use. Above it the chunk's queries are
/// split into blocks of a multiple of 32 rows, each with its own mask and
/// blk, rebuilt per layer and drained before the next block allocates.
pub(crate) const QWEN_PREFILL_TENSOR_MASK_CAP_BYTES: u64 = 256 << 20;

/// Head dim of every Qwen3.5/3.6/3.8 full-attention layer.
const HEAD_DIM: u32 = 256;

/// Resolved resumed-prefill attention route.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum QwenPrefillAttention {
    /// Route (c): F32 dequant + undo, BF16 casts, BF16 resume kernel.
    #[default]
    Bridge,
    /// Route (c′): F16 rotated staging + tensor flash attention.
    Tensor,
    /// The switch is on but the device lacks the Metal tensor API: route (c).
    BridgeTensorUnavailable,
}

impl QwenPrefillAttention {
    /// Wire/banner label.
    pub fn label(self) -> &'static str {
        match self {
            Self::Bridge => "bridge",
            Self::Tensor => "tensor",
            Self::BridgeTensorUnavailable => "bridge (tensor unavailable)",
        }
    }

    pub fn uses_tensor(self) -> bool {
        self == Self::Tensor
    }
}

/// Parse [`QWEN_PREFILL_TENSOR_ATTN_ENV`]; unset or blank means off.
pub(crate) fn parse_prefill_tensor_attn(raw: Option<&str>) -> Result<bool, String> {
    let Some(raw) = raw.map(str::trim).filter(|raw| !raw.is_empty()) else {
        return Ok(false);
    };
    match raw.to_ascii_lowercase().as_str() {
        "1" | "on" | "true" => Ok(true),
        "0" | "off" | "false" => Ok(false),
        _ => Err(format!(
            "{QWEN_PREFILL_TENSOR_ATTN_ENV} must be 1/on/true or 0/off/false, got {raw:?}"
        )),
    }
}

/// Resolve the route from the request and the tensor-API probe. The probe
/// runs only when the tensor route is requested; `Ok(false)` falls back to
/// the bridge, any other probe failure is an error.
pub(crate) fn select_prefill_attention(
    requested: bool,
    probe: impl FnOnce() -> mlx_native::Result<bool>,
) -> Result<QwenPrefillAttention> {
    if !requested {
        return Ok(QwenPrefillAttention::Bridge);
    }
    match probe().context("Qwen resumed-prefill tensor attention probe")? {
        true => Ok(QwenPrefillAttention::Tensor),
        false => Ok(QwenPrefillAttention::BridgeTensorUnavailable),
    }
}

/// Read [`QWEN_PREFILL_TENSOR_ATTN_ENV`] once (an invalid value warns and
/// keeps the bridge), probe `flash_attn_prefill_tensor_f16_d256` when
/// requested, and log the resolved route.
pub fn resolve_prefill_attention_from_process() -> Result<QwenPrefillAttention> {
    let raw = std::env::var(QWEN_PREFILL_TENSOR_ATTN_ENV).ok();
    let requested = parse_prefill_tensor_attn(raw.as_deref()).unwrap_or_else(|error| {
        tracing::warn!(
            error,
            fallback = false,
            "invalid Qwen prefill tensor attention switch"
        );
        false
    });
    let resolved = select_prefill_attention(requested, || {
        let device = MlxDevice::new()?;
        let mut registry = KernelRegistry::new();
        tensor::flash_attn_prefill_tensor_available(&mut registry, &device, HEAD_DIM)
    })?;
    tracing::info!(
        requested,
        qwen_prefill_attention = resolved.label(),
        "Qwen resumed-prefill attention route"
    );
    Ok(resolved)
}

thread_local! {
    static ACTIVE_CHUNK: RefCell<Option<TensorResumeChunk>> = const { RefCell::new(None) };
}

/// Per-forward scope that enables route (c′) on this thread and owns the
/// chunk's shared staging, mask and blk until the forward returns (after its
/// terminal wait). A `Bridge` route opens an inert scope.
pub(crate) struct TensorResumeScope {
    previous: Option<TensorResumeChunk>,
    active: bool,
}

impl TensorResumeScope {
    pub(crate) fn enter(route: QwenPrefillAttention) -> Self {
        if !route.uses_tensor() {
            return Self {
                previous: None,
                active: false,
            };
        }
        let previous =
            ACTIVE_CHUNK.with(|chunk| chunk.borrow_mut().replace(TensorResumeChunk::default()));
        Self {
            previous,
            active: true,
        }
    }
}

impl Drop for TensorResumeScope {
    fn drop(&mut self) {
        if self.active {
            let previous = self.previous.take();
            ACTIVE_CHUNK.with(|chunk| *chunk.borrow_mut() = previous);
        }
    }
}

/// Run `f` on the active chunk state; `None` when no tensor scope is open.
pub(crate) fn with_active_chunk<R>(f: impl FnOnce(&mut TensorResumeChunk) -> R) -> Option<R> {
    ACTIVE_CHUNK.with(|chunk| chunk.borrow_mut().as_mut().map(f))
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct ChunkShape {
    cur_len: u32,
    seq_len: u32,
    n_heads: u32,
    n_kv_heads: u32,
}

struct ChunkBuffers {
    shape: ChunkShape,
    kv_pad: u32,
    /// Query rows per tensor dispatch; `seq_len` unless the mask cap splits.
    block_rows: u32,
    /// F16 `(n_kv_heads − 1)·L + kv_pad` rows of `head_dim`, rewritten by
    /// every layer's dequant. Not pooled: rows past `n_kv_heads·L` are read
    /// (masked) by the kernel and must stay finite, so they are zeroed once
    /// here and never written again; pooled storage may hold any bit
    /// pattern, and the pool is reset between prefill layers.
    k_stage: MlxBuffer,
    v_stage: MlxBuffer,
    /// Mask and blk shared by every FA layer when `block_rows == seq_len`.
    shared_mask_blk: Option<(MlxBuffer, MlxBuffer)>,
}

/// Chunk state shared by the FA layers of one resumed-prefill chunk.
#[derive(Default)]
pub struct TensorResumeChunk {
    buffers: Option<ChunkBuffers>,
}

/// Rows per query block so that one block's mask fits `cap_bytes`.
fn query_block_rows(seq_len: u32, kv_pad: u32, cap_bytes: u64) -> u32 {
    let row_bytes = u64::from(kv_pad) * 2;
    if u64::from(seq_len) * row_bytes <= cap_bytes {
        return seq_len;
    }
    let bq = u64::from(tensor::queries_per_threadgroup(HEAD_DIM).unwrap_or(32));
    let rows = (cap_bytes / row_bytes) / bq * bq;
    rows.max(bq).min(u64::from(seq_len)) as u32
}

fn alloc_stage(
    device: &MlxDevice,
    n_kv_heads: usize,
    kv_seq_len: usize,
    kv_pad: usize,
    what: &str,
) -> Result<MlxBuffer> {
    let hd = HEAD_DIM as usize;
    let written = n_kv_heads * kv_seq_len * hd;
    let elems = written + (kv_pad - kv_seq_len) * hd;
    let mut buffer = device
        .alloc_buffer(elems * 2, DType::F16, vec![elems])
        .map_err(|e| anyhow!("tensor resume {what} staging alloc: {e}"))?;
    buffer
        .as_mut_slice::<u16>()
        .map_err(|e| anyhow!("tensor resume {what} staging zero fill: {e}"))?[written..]
        .fill(0);
    Ok(buffer)
}

/// Mask `[rows, kv_pad]` for queries at absolute `q_abs_offset..` and its
/// blk at `(bq, bk) = (32, 64)`.
fn encode_mask_blk(
    enc: &mut CommandEncoder,
    device: &MlxDevice,
    registry: &mut KernelRegistry,
    rows: u32,
    kv_pad: u32,
    q_abs_offset: u32,
) -> Result<(MlxBuffer, MlxBuffer)> {
    let mask = build_sdpa_mask_f16(
        device,
        registry,
        enc,
        &SdpaMaskParams {
            seq_len_q: rows,
            seq_len_k: kv_pad,
            window_size: None,
            causal: true,
            q_abs_offset,
        },
    )
    .map_err(|e| anyhow!("tensor resume mask: {e}"))?;
    // The builder's write is not declared to the encoder: fence before the
    // blk classifier reads it.
    enc.memory_barrier();
    let params = BlkParams {
        seq_len_q: rows,
        seq_len_k: kv_pad,
        bq: tensor::queries_per_threadgroup(HEAD_DIM).map_err(|e| anyhow!("{e}"))?,
        bk: tensor::CACHE_ROWS_PER_STEP,
    };
    let blk =
        alloc_blk_buffer(device, &params).map_err(|e| anyhow!("tensor resume blk alloc: {e}"))?;
    dispatch_flash_attn_prefill_blk_f16(enc, device, registry, &mask, &blk, &params)
        .map_err(|e| anyhow!("tensor resume blk: {e}"))?;
    Ok((mask, blk))
}

#[allow(clippy::too_many_arguments)]
fn dispatch_tensor(
    enc: &mut CommandEncoder,
    registry: &mut KernelRegistry,
    device: &MlxDevice,
    (q, out): (&MlxBuffer, &MlxBuffer),
    (k, v): (&MlxBuffer, &MlxBuffer),
    (mask, blk): (&MlxBuffer, &MlxBuffer),
    rows: u32,
    shape: ChunkShape,
    kv_seq_len: u32,
    kv_pad: u32,
) -> Result<()> {
    tensor::dispatch_flash_attn_prefill_tensor_f16(
        enc,
        registry,
        device,
        q,
        k,
        v,
        mask,
        blk,
        out,
        &tensor::FlashAttnPrefillTensorParams {
            rows,
            n_heads: shape.n_heads,
            n_kv_heads: shape.n_kv_heads,
            head_dim: HEAD_DIM,
            kv_pad,
            q_row_stride: shape.n_heads * HEAD_DIM,
            q_head_stride: HEAD_DIM,
            k_row_stride: HEAD_DIM,
            k_head_stride: kv_seq_len * HEAD_DIM,
            v_row_stride: HEAD_DIM,
            v_head_stride: kv_seq_len * HEAD_DIM,
            scale: 1.0 / (HEAD_DIM as f32).sqrt(),
        },
    )
    .map_err(|e| anyhow!("tensor resume flash attention: {e}"))
}

/// Route (c′) for one FA layer of a resumed chunk. `q_seq_major` (F32
/// `[seq_len, n_heads, 256]`, `q_rope`) is rotated in place and must have no
/// reader after attention. Returns F32 `[seq_len, n_heads, 256]` seq-major,
/// committed without a host wait; queue order orders the consumer.
#[allow(clippy::too_many_arguments)]
pub(crate) fn apply_tq_prefill_seq_major_resume_tensor_for_slot(
    device: &MlxDevice,
    registry: &mut KernelRegistry,
    slot: &FullAttnKvSlot,
    q_seq_major: &MlxBuffer,
    seq_len: u32,
    cur_len: u32,
    kv_seq_len: u32,
    cache_capacity: u32,
    n_heads: u32,
    n_kv_heads: u32,
    head_dim: u32,
    slot_id: SlotId,
    chunk: &mut TensorResumeChunk,
) -> Result<MlxBuffer> {
    encode_tensor_resume(
        device,
        registry,
        slot,
        q_seq_major,
        (seq_len, cur_len, kv_seq_len, cache_capacity),
        (n_heads, n_kv_heads, head_dim),
        slot_id,
        chunk,
        QWEN_PREFILL_TENSOR_MASK_CAP_BYTES,
    )
}

#[allow(clippy::too_many_arguments)]
fn encode_tensor_resume(
    device: &MlxDevice,
    registry: &mut KernelRegistry,
    slot: &FullAttnKvSlot,
    q_seq_major: &MlxBuffer,
    (seq_len, cur_len, kv_seq_len, cache_capacity): (u32, u32, u32, u32),
    (n_heads, n_kv_heads, head_dim): (u32, u32, u32),
    slot_id: SlotId,
    chunk: &mut TensorResumeChunk,
    mask_cap_bytes: u64,
) -> Result<MlxBuffer> {
    ensure!(
        head_dim == HEAD_DIM,
        "tensor resume: head_dim must be {HEAD_DIM}, got {head_dim}"
    );
    ensure!(
        seq_len > 0 && cur_len.checked_add(seq_len) == Some(kv_seq_len),
        "tensor resume: cur_len ({cur_len}) + seq_len ({seq_len}) != kv_seq_len ({kv_seq_len})"
    );
    ensure!(
        n_kv_heads > 0 && n_heads % n_kv_heads == 0,
        "tensor resume: n_heads {n_heads} is not a multiple of n_kv_heads {n_kv_heads}"
    );
    ensure!(
        slot.tq.is_some(),
        "tensor resume requires a TQ-active full-attention slot"
    );
    let shape = ChunkShape {
        cur_len,
        seq_len,
        n_heads,
        n_kv_heads,
    };
    if let Some(built) = chunk.buffers.as_ref() {
        ensure!(
            built.shape == shape,
            "tensor resume: layer shape {shape:?} differs from the chunk's {:?}",
            built.shape
        );
    } else {
        let kv_pad = tensor::kv_pad(kv_seq_len);
        let (nkv, l, pad) = (n_kv_heads as usize, kv_seq_len as usize, kv_pad as usize);
        chunk.buffers = Some(ChunkBuffers {
            shape,
            kv_pad,
            block_rows: query_block_rows(seq_len, kv_pad, mask_cap_bytes),
            k_stage: alloc_stage(device, nkv, l, pad, "K")?,
            v_stage: alloc_stage(device, nkv, l, pad, "V")?,
            shared_mask_blk: None,
        });
    }
    let bufs = chunk.buffers.as_mut().expect("chunk buffers built above");
    let kv_pad = bufs.kv_pad;
    let row_elems = (n_heads * HEAD_DIM) as usize;
    let out_elems = seq_len as usize * row_elems;
    let out = device
        .alloc_buffer(
            out_elems * 4,
            DType::F32,
            vec![seq_len as usize, n_heads as usize, HEAD_DIM as usize],
        )
        .map_err(|e| anyhow!("tensor resume output alloc: {e}"))?;
    let q_rows = seq_len
        .checked_mul(n_heads)
        .ok_or_else(|| anyhow!("tensor resume: seq_len × n_heads overflows u32"))?;

    let mut enc = device
        .command_encoder()
        .map(crate::inference::native_observation::encoder)
        .context("tensor resume encoder")?;
    for (is_k, stage) in [(true, &bufs.k_stage), (false, &bufs.v_stage)] {
        slot.dequant_seq_to_f16_rotated_into_for_slot(
            is_k,
            kv_seq_len,
            cache_capacity,
            n_kv_heads,
            HEAD_DIM,
            slot_id,
            stage,
            &mut enc,
            registry,
            device,
        )?;
    }
    dispatch_fwht_sign_premult_f32(
        &mut enc,
        registry,
        device.metal_device(),
        q_seq_major,
        q_rows,
        HEAD_DIM,
    )
    .map_err(|e| anyhow!("tensor resume Q premult: {e}"))?;

    let stages = (&bufs.k_stage, &bufs.v_stage);
    if bufs.block_rows == seq_len {
        if bufs.shared_mask_blk.is_none() {
            bufs.shared_mask_blk = Some(encode_mask_blk(
                &mut enc, device, registry, seq_len, kv_pad, cur_len,
            )?);
        }
        let (mask, blk) = bufs
            .shared_mask_blk
            .as_ref()
            .expect("shared mask built above");
        enc.memory_barrier();
        dispatch_tensor(
            &mut enc,
            registry,
            device,
            (q_seq_major, &out),
            stages,
            (mask, blk),
            seq_len,
            shape,
            kv_seq_len,
            kv_pad,
        )?;
    } else {
        // Mask-cap split: one block's mask is live at a time, so each block
        // but the last drains before the next allocates its mask.
        let mut start = 0u32;
        while start < seq_len {
            let rows = bufs.block_rows.min(seq_len - start);
            if start > 0 {
                enc.commit_and_wait_labeled("layer.full_attn.tensor_resume_block")
                    .context("tensor resume query block")?;
                enc = device
                    .command_encoder()
                    .map(crate::inference::native_observation::encoder)
                    .context("tensor resume block encoder")?;
            }
            let (mask, blk) =
                encode_mask_blk(&mut enc, device, registry, rows, kv_pad, cur_len + start)?;
            enc.memory_barrier();
            let offset = (start as usize * row_elems * 4) as u64;
            let block_elems = rows as usize * row_elems;
            dispatch_tensor(
                &mut enc,
                registry,
                device,
                (
                    &q_seq_major.slice_view(offset, block_elems),
                    &out.slice_view(offset, block_elems),
                ),
                stages,
                (&mask, &blk),
                rows,
                shape,
                kv_seq_len,
                kv_pad,
            )?;
            start += rows;
        }
    }
    enc.memory_barrier();
    dispatch_fwht_sign_undo_f32(
        &mut enc,
        registry,
        device.metal_device(),
        &out,
        q_rows,
        HEAD_DIM,
    )
    .map_err(|e| anyhow!("tensor resume output undo: {e}"))?;
    // No host wait: the ops6-7 command buffer consumes `out` in queue order,
    // and the scope keeps the staging, mask and blk alive past the forward's
    // terminal wait.
    enc.commit_labeled("layer.full_attn.tensor_resume");
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::inference::models::qwen35::kv_cache::HybridKvCache;
    use crate::inference::models::qwen35::{default_layer_types, Qwen35Config, Qwen35Variant};
    use rand::{rngs::StdRng, Rng, SeedableRng};
    use rayon::prelude::*;

    const NH: u32 = 24;
    const NKV: u32 = 4;
    const D: usize = HEAD_DIM as usize;

    /// Class B bound for route (c′) against the F64 reference over the same
    /// dequantized K/V: `max|Δ| ≤ K_F16 · 2⁻¹¹ · max|V|`. Q (after the
    /// power-of-two scale), K, V and P are rounded to half (relative error
    /// ≤ 2⁻¹¹ each). The output is a convex combination of V rows, so the V
    /// and P roundings contribute ≤ 2⁻¹¹·max|V| each; the Q and K roundings
    /// move each score by ≤ 2·2⁻¹¹·|s| (|s| ≲ 2 for this data), which moves
    /// the probabilities by about that relative amount. The FWHT rotations
    /// and F32 accumulation add ~1e-6 relative. That sums to ≈ 6·2⁻¹¹; k = 8
    /// is the Gemma F1 gate's constant (3.9e-3·max|V|).
    const K_F16: f64 = 8.0;
    /// Same reasoning at BF16 (8-bit mantissa) for route (c).
    const K_BF16: f64 = 8.0;
    const EPS_F16: f64 = 1.0 / 2048.0;
    const EPS_BF16: f64 = 1.0 / 256.0;

    fn qwen38_attn_cfg() -> Qwen35Config {
        Qwen35Config {
            variant: Qwen35Variant::Dense,
            hidden_size: 5120,
            num_hidden_layers: 4,
            num_attention_heads: NH,
            num_key_value_heads: NKV,
            head_dim: HEAD_DIM,
            linear_num_key_heads: 16,
            linear_num_value_heads: 48,
            linear_key_head_dim: 128,
            linear_value_head_dim: 128,
            linear_conv_kernel_dim: 4,
            full_attention_interval: 4,
            layer_types: default_layer_types(4, 4),
            partial_rotary_factor: 0.25,
            rope_theta: 1e7,
            rotary_dim: 64,
            mrope_section: [11, 11, 10, 0],
            mrope_interleaved: true,
            rms_norm_eps: 1e-6,
            max_position_embeddings: 262144,
            vocab_size: 248320,
            attn_output_gate: true,
            mtp_num_hidden_layers: 0,
            mtp_use_dedicated_embeddings: true,
            intermediate_size: Some(17408),
            moe: None,
        }
    }

    fn codebook_bits() -> u32 {
        let bits = crate::debug::INVESTIGATION_ENV.tq_codebook_bits;
        if matches!(bits, 5 | 6 | 8) {
            bits
        } else {
            8
        }
    }

    fn upload(device: &MlxDevice, data: &[f32], shape: Vec<usize>) -> MlxBuffer {
        let mut buffer = device
            .alloc_buffer(data.len() * 4, DType::F32, shape)
            .expect("alloc upload");
        buffer
            .as_mut_slice::<f32>()
            .expect("upload map")
            .copy_from_slice(data);
        buffer
    }

    fn drain(device: &MlxDevice) {
        device
            .command_encoder()
            .expect("drain encoder")
            .commit_and_wait()
            .expect("drain");
    }

    /// One TQ-only slot holding `total` positions of random K/V written by
    /// the production `hadamard_quantize` encoder, the un-rotated F32
    /// dequant of that cache (the reference's K/V) and `queries` rows of
    /// random Q.
    struct Fixture {
        device: MlxDevice,
        registry: KernelRegistry,
        cache: HybridKvCache,
        capacity: u32,
        /// `[NKV, total, D]` un-rotated F32 dequant.
        k_ref: Vec<f32>,
        v_ref: Vec<f32>,
        total: u32,
        /// `[queries, NH, D]`.
        q: Vec<f32>,
    }

    impl Fixture {
        fn new(total: u32, queries: u32, seed: u64) -> Option<Self> {
            let device = match MlxDevice::new() {
                Ok(device) => device,
                Err(e) => {
                    eprintln!("skipping: no Metal device: {e}");
                    return None;
                }
            };
            let mut registry = KernelRegistry::new();
            mlx_native::ops::flash_attn_prefill::register(&mut registry);
            let capacity = total;
            let cfg = qwen38_attn_cfg();
            let mut cache = HybridKvCache::new_with_options(&cfg, &device, capacity, 1, true)
                .expect("TQ cache");
            let mut rng = StdRng::seed_from_u64(seed);
            let kv_elems = total as usize * (NKV as usize) * D;
            let k: Vec<f32> = (0..kv_elems).map(|_| rng.gen_range(-1.0f32..1.0)).collect();
            let v: Vec<f32> = (0..kv_elems).map(|_| rng.gen_range(-1.0f32..1.0)).collect();
            let q: Vec<f32> = (0..queries as usize * NH as usize * D)
                .map(|_| rng.gen_range(-2.0f32..2.0))
                .collect();
            let shape = vec![total as usize, NKV as usize, D];
            let (k_gpu, v_gpu) = (
                upload(&device, &k, shape.clone()),
                upload(&device, &v, shape),
            );
            let slot = &mut cache.full_attn[0];
            let mut enc = device.command_encoder().expect("encode encoder");
            for (is_k, src) in [(true, &k_gpu), (false, &v_gpu)] {
                slot.encode_seq_tokens_to_tq(
                    src,
                    is_k,
                    total,
                    NKV,
                    HEAD_DIM,
                    capacity,
                    0,
                    0,
                    false,
                    1.0,
                    codebook_bits(),
                    &mut enc,
                    &mut registry,
                    &device,
                )
                .expect("TQ encode");
            }
            enc.commit_and_wait().expect("TQ encode wait");
            let slot = &cache.full_attn[0];
            let mut enc = device.command_encoder().expect("dequant encoder");
            let deq = |is_k, enc: &mut CommandEncoder, registry: &mut KernelRegistry| {
                slot.dequant_seq_to_temp_f32_unrotated_for_slot(
                    is_k,
                    total,
                    0,
                    capacity,
                    NKV,
                    HEAD_DIM,
                    SlotId(0),
                    enc,
                    registry,
                    &device,
                )
                .expect("reference dequant")
            };
            let k_buf = deq(true, &mut enc, &mut registry);
            let v_buf = deq(false, &mut enc, &mut registry);
            enc.commit_and_wait().expect("reference dequant wait");
            let k_ref = k_buf.as_slice::<f32>().expect("k ref").to_vec();
            let v_ref = v_buf.as_slice::<f32>().expect("v ref").to_vec();
            Some(Self {
                device,
                registry,
                cache,
                capacity,
                k_ref,
                v_ref,
                total,
                q,
            })
        }

        fn q_rows(&self, first: u32, rows: u32) -> &[f32] {
            let row = NH as usize * D;
            &self.q[first as usize * row..(first + rows) as usize * row]
        }

        /// Route (c′) for queries `first..first + rows` of the fixture's Q at
        /// absolute positions `cur_len..`, drained.
        fn tensor(
            &mut self,
            cur_len: u32,
            first: u32,
            rows: u32,
            chunk: &mut TensorResumeChunk,
            mask_cap_bytes: u64,
        ) -> Result<Vec<f32>> {
            let q = upload(
                &self.device,
                self.q_rows(first, rows),
                vec![rows as usize, NH as usize, D],
            );
            let out = encode_tensor_resume(
                &self.device,
                &mut self.registry,
                &self.cache.full_attn[0],
                &q,
                (rows, cur_len, cur_len + rows, self.capacity),
                (NH, NKV, HEAD_DIM),
                SlotId(0),
                chunk,
                mask_cap_bytes,
            )?;
            drain(&self.device);
            Ok(out.as_slice::<f32>().expect("tensor out").to_vec())
        }

        /// Route (c), the shipped bridge, on the same input.
        fn bridge(&mut self, cur_len: u32, first: u32, rows: u32) -> Vec<f32> {
            let q = upload(
                &self.device,
                self.q_rows(first, rows),
                vec![rows as usize, NH as usize, D],
            );
            let out = super::super::gpu_full_attn::apply_flash_attn_prefill_seq_major_resume_via_tq_cache_for_slot(
                &self.device,
                &mut self.registry,
                &self.cache.full_attn[0],
                &q,
                rows,
                cur_len,
                cur_len + rows,
                self.capacity,
                NH,
                NKV,
                HEAD_DIM,
                SlotId(0),
            )
            .expect("bridge");
            out.as_slice::<f32>().expect("bridge out").to_vec()
        }

        /// CPU F64 causal attention of queries `first..first + rows` at
        /// absolute positions `cur_len..` over the un-rotated dequant.
        fn reference(&self, cur_len: u32, first: u32, rows: u32) -> Vec<f64> {
            let q = self.q_rows(first, rows);
            let total = self.total as usize;
            let group = (NH / NKV) as usize;
            let scale = 1.0 / (D as f64).sqrt();
            let mut out = vec![0f64; rows as usize * NH as usize * D];
            out.par_chunks_mut(D).enumerate().for_each(|(index, dst)| {
                let (r, h) = (index / NH as usize, index % NH as usize);
                let kvh = h / group;
                let qv = &q[index * D..(index + 1) * D];
                let visible = cur_len as usize + r + 1;
                let mut scores = Vec::with_capacity(visible);
                for p in 0..visible {
                    let kv = &self.k_ref[(kvh * total + p) * D..(kvh * total + p + 1) * D];
                    let dot: f64 = qv.iter().zip(kv).map(|(a, b)| *a as f64 * *b as f64).sum();
                    scores.push(dot * scale);
                }
                let max = scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                let mut sum = 0.0;
                for s in &mut scores {
                    *s = (*s - max).exp();
                    sum += *s;
                }
                for (p, w) in scores.iter().enumerate() {
                    let vv = &self.v_ref[(kvh * total + p) * D..(kvh * total + p + 1) * D];
                    for (o, x) in dst.iter_mut().zip(vv) {
                        *o += w / sum * *x as f64;
                    }
                }
            });
            out
        }

        fn max_abs_v(&self, kv_len: u32) -> f64 {
            let total = self.total as usize;
            (0..NKV as usize)
                .flat_map(|h| self.v_ref[h * total * D..(h * total + kv_len as usize) * D].iter())
                .fold(0f64, |m, x| m.max((*x as f64).abs()))
        }
    }

    fn max_abs_diff_f64(a: &[f32], reference: &[f64]) -> f64 {
        assert_eq!(a.len(), reference.len());
        a.iter()
            .zip(reference)
            .map(|(x, r)| {
                assert!(x.is_finite(), "non-finite output");
                (*x as f64 - r).abs()
            })
            .fold(0.0, f64::max)
    }

    fn max_abs_diff(a: &[f32], b: &[f32]) -> f64 {
        assert_eq!(a.len(), b.len());
        a.iter()
            .zip(b)
            .map(|(x, y)| (*x as f64 - *y as f64).abs())
            .fold(0.0, f64::max)
    }

    #[test]
    fn prefill_tensor_attn_switch_parses_and_defaults_off() {
        assert_eq!(parse_prefill_tensor_attn(None), Ok(false));
        assert_eq!(parse_prefill_tensor_attn(Some("  ")), Ok(false));
        for raw in ["1", "on", "true", " TRUE ", "On"] {
            assert_eq!(parse_prefill_tensor_attn(Some(raw)), Ok(true), "{raw:?}");
        }
        for raw in ["0", "off", "false", "OFF"] {
            assert_eq!(parse_prefill_tensor_attn(Some(raw)), Ok(false), "{raw:?}");
        }
        for raw in ["2", "tensor", "yes"] {
            let error = parse_prefill_tensor_attn(Some(raw)).unwrap_err();
            assert!(error.contains(QWEN_PREFILL_TENSOR_ATTN_ENV), "{error}");
        }
    }

    #[test]
    fn prefill_tensor_attn_selection_probes_only_when_requested() {
        let off = select_prefill_attention(false, || panic!("probed while off")).unwrap();
        assert_eq!(off, QwenPrefillAttention::Bridge);
        assert_eq!(
            select_prefill_attention(true, || Ok(true)).unwrap(),
            QwenPrefillAttention::Tensor
        );
        assert_eq!(
            select_prefill_attention(true, || Ok(false)).unwrap(),
            QwenPrefillAttention::BridgeTensorUnavailable
        );
        assert!(select_prefill_attention(true, || {
            Err(mlx_native::MlxError::InvalidArgument("probe".into()))
        })
        .is_err());
        assert_eq!(QwenPrefillAttention::default().label(), "bridge");
        assert_eq!(QwenPrefillAttention::Tensor.label(), "tensor");
        assert_eq!(
            QwenPrefillAttention::BridgeTensorUnavailable.label(),
            "bridge (tensor unavailable)"
        );
    }

    #[test]
    fn prefill_tensor_scope_enables_only_the_tensor_route() {
        assert!(with_active_chunk(|_| ()).is_none());
        for route in [
            QwenPrefillAttention::Bridge,
            QwenPrefillAttention::BridgeTensorUnavailable,
        ] {
            let _scope = TensorResumeScope::enter(route);
            assert!(with_active_chunk(|_| ()).is_none(), "{route:?}");
        }
        {
            let _scope = TensorResumeScope::enter(QwenPrefillAttention::Tensor);
            assert!(with_active_chunk(|_| ()).is_some());
        }
        assert!(with_active_chunk(|_| ()).is_none());
    }

    #[test]
    fn prefill_tensor_query_blocks_respect_the_mask_cap() {
        assert_eq!(
            query_block_rows(512, 2560, QWEN_PREFILL_TENSOR_MASK_CAP_BYTES),
            512
        );
        // 512 × 262144 × 2 B is exactly the cap.
        assert_eq!(
            query_block_rows(512, 262_144, QWEN_PREFILL_TENSOR_MASK_CAP_BYTES),
            512
        );
        // 2048 rows at kv_pad 131072 need 512 MiB: blocks of 1024 rows.
        assert_eq!(
            query_block_rows(2048, 131_072, QWEN_PREFILL_TENSOR_MASK_CAP_BYTES),
            1024
        );
        // A cap below one 32-row block still dispatches 32 rows.
        assert_eq!(query_block_rows(512, 1024, 1), 32);
        assert_eq!(query_block_rows(100, 1024, 64 * 1024 * 2), 64);
    }

    /// Gate 1 (Class B): route (c′) against CPU F64 attention over the same
    /// dequantized K/V at Qwen3.8 shapes, `max|Δ| ≤ 8·2⁻¹¹·max|V|`.
    /// `(100, 64)` and `(64, 33)` put `L % 64 ≠ 0`, so the padded tail and the
    /// cross-head rows are read (masked).
    #[test]
    fn prefill_tensor_resume_matches_f64_reference_at_qwen38_shapes() {
        let _gpu = crate::inference::hf2q_gpu_test_lock();
        for (cur_len, seq_len) in [
            (64, 64),
            (64, 512),
            (2048, 64),
            (2048, 512),
            (100, 64),
            (64, 33),
        ] {
            let Some(mut fx) = Fixture::new(cur_len + seq_len, seq_len, 0xF3 + cur_len as u64)
            else {
                return;
            };
            let mut chunk = TensorResumeChunk::default();
            let out = fx
                .tensor(
                    cur_len,
                    0,
                    seq_len,
                    &mut chunk,
                    QWEN_PREFILL_TENSOR_MASK_CAP_BYTES,
                )
                .expect("tensor resume");
            let reference = fx.reference(cur_len, 0, seq_len);
            let max_v = fx.max_abs_v(cur_len + seq_len);
            let delta = max_abs_diff_f64(&out, &reference);
            let bound = K_F16 * EPS_F16 * max_v;
            eprintln!(
                "[F3 F64] c={cur_len} n={seq_len} max|Δ|={delta:.3e} bound={bound:.3e} max|V|={max_v:.3}"
            );
            assert!(
                delta <= bound,
                "c={cur_len} n={seq_len}: max|Δ| {delta:.3e} > {bound:.3e}"
            );
        }
    }

    /// Gate 2 (Class A): repeated runs are bitwise identical, including the
    /// mask/blk reuse a later layer of the same chunk takes.
    #[test]
    fn prefill_tensor_resume_repeats_bitwise() {
        let _gpu = crate::inference::hf2q_gpu_test_lock();
        let (cur_len, seq_len) = (2048, 512);
        let Some(mut fx) = Fixture::new(cur_len + seq_len, seq_len, 7) else {
            return;
        };
        let mut chunk = TensorResumeChunk::default();
        let cap = QWEN_PREFILL_TENSOR_MASK_CAP_BYTES;
        let first = fx
            .tensor(cur_len, 0, seq_len, &mut chunk, cap)
            .expect("first");
        let shared = fx
            .tensor(cur_len, 0, seq_len, &mut chunk, cap)
            .expect("shared mask");
        let fresh = fx
            .tensor(cur_len, 0, seq_len, &mut TensorResumeChunk::default(), cap)
            .expect("fresh chunk");
        assert!(first
            .iter()
            .zip(&shared)
            .all(|(a, b)| a.to_bits() == b.to_bits()));
        assert!(first
            .iter()
            .zip(&fresh)
            .all(|(a, b)| a.to_bits() == b.to_bits()));
    }

    /// Gate 3 (Class A): a resumed suffix split into pieces of more than 32
    /// queries (1×1024, 2×512, 512+33+479), or into 64-row query blocks by
    /// the mask cap, gives bitwise-identical rows.
    #[test]
    fn prefill_tensor_resume_rows_are_chunking_invariant() {
        let _gpu = crate::inference::hf2q_gpu_test_lock();
        let (prefix, suffix) = (100u32, 1024u32);
        let Some(mut fx) = Fixture::new(prefix + suffix, suffix, 11) else {
            return;
        };
        let cap = QWEN_PREFILL_TENSOR_MASK_CAP_BYTES;
        let run = |fx: &mut Fixture, pieces: &[u32], cap: u64| -> Vec<f32> {
            let mut rows = Vec::new();
            let mut first = 0;
            for &piece in pieces {
                let mut chunk = TensorResumeChunk::default();
                rows.extend(
                    fx.tensor(prefix + first, first, piece, &mut chunk, cap)
                        .expect("piece"),
                );
                first += piece;
            }
            rows
        };
        let whole = run(&mut fx, &[1024], cap);
        let kv_pad = u64::from(tensor::kv_pad(prefix + suffix));
        for (label, rows) in [
            ("2x512", run(&mut fx, &[512, 512], cap)),
            ("512+33+479", run(&mut fx, &[512, 33, 479], cap)),
            (
                "64-row query blocks",
                run(&mut fx, &[1024], 64 * kv_pad * 2),
            ),
        ] {
            let differing = whole
                .iter()
                .zip(&rows)
                .filter(|(a, b)| a.to_bits() != b.to_bits())
                .count();
            assert_eq!(
                differing, 0,
                "{label}: {differing} elements differ from 1x1024"
            );
        }
    }

    /// Route (c′) and the shipped route (c) agree within the sum of their
    /// Class B bounds on the same input; distances to the F64 reference are
    /// printed as a diagnostic.
    #[test]
    fn prefill_tensor_resume_agrees_with_bridge_route() {
        let _gpu = crate::inference::hf2q_gpu_test_lock();
        for (cur_len, seq_len) in [(64, 64), (2048, 512)] {
            let Some(mut fx) = Fixture::new(cur_len + seq_len, seq_len, 0xB0 + cur_len as u64)
            else {
                return;
            };
            let mut chunk = TensorResumeChunk::default();
            let tensor_out = fx
                .tensor(
                    cur_len,
                    0,
                    seq_len,
                    &mut chunk,
                    QWEN_PREFILL_TENSOR_MASK_CAP_BYTES,
                )
                .expect("tensor");
            let bridge_out = fx.bridge(cur_len, 0, seq_len);
            let max_v = fx.max_abs_v(cur_len + seq_len);
            let delta = max_abs_diff(&tensor_out, &bridge_out);
            let bound = (K_F16 * EPS_F16 + K_BF16 * EPS_BF16) * max_v;
            let reference = fx.reference(cur_len, 0, seq_len);
            eprintln!(
                "[F3 vs bridge] c={cur_len} n={seq_len} max|Δ|={delta:.3e} bound={bound:.3e} \
                 tensor→F64={:.3e} bridge→F64={:.3e}",
                max_abs_diff_f64(&tensor_out, &reference),
                max_abs_diff_f64(&bridge_out, &reference)
            );
            assert!(
                delta <= bound,
                "c={cur_len} n={seq_len}: {delta:.3e} > {bound:.3e}"
            );
        }
    }

    /// Later FA layers of a chunk must see the chunk's `(cur_len, seq_len)`.
    #[test]
    fn prefill_tensor_resume_rejects_a_shape_change_within_a_chunk() {
        let _gpu = crate::inference::hf2q_gpu_test_lock();
        let Some(mut fx) = Fixture::new(256, 128, 3) else {
            return;
        };
        let cap = QWEN_PREFILL_TENSOR_MASK_CAP_BYTES;
        let mut chunk = TensorResumeChunk::default();
        fx.tensor(128, 0, 128, &mut chunk, cap)
            .expect("first layer");
        let error = fx.tensor(192, 0, 64, &mut chunk, cap).unwrap_err();
        assert!(
            format!("{error:#}").contains("differs from the chunk"),
            "{error:#}"
        );
    }

    /// Micro-bench (ignored): µs per FA layer of route (c′) and route (c)
    /// at c ∈ {2048, 8192}, n = 512, Qwen3.8 shapes, comparable to
    /// `bench_qwen38_prefill_ops` `fa_resume_*`. Each sample runs 16 layers
    /// of one chunk (mask/blk built in the first) and drains once.
    /// `cargo test --release --bin hf2q prefill_tensor_resume_bench -- --ignored --nocapture`
    #[test]
    #[ignore = "timing micro-bench; run explicitly"]
    fn prefill_tensor_resume_bench() {
        let _gpu = crate::inference::hf2q_gpu_test_lock();
        const LAYERS: usize = 16;
        const SAMPLES: usize = 5;
        let seq_len = 512u32;
        for cur_len in [2048u32, 8192] {
            let Some(mut fx) = Fixture::new(cur_len + seq_len, seq_len, 5) else {
                return;
            };
            let q = fx.q_rows(0, seq_len).to_vec();
            let shape = vec![seq_len as usize, NH as usize, D];
            let time = |fx: &mut Fixture, tensor_route: bool| -> Vec<f64> {
                let mut samples = Vec::new();
                for _ in 0..=SAMPLES {
                    let qs: Vec<MlxBuffer> = (0..LAYERS)
                        .map(|_| upload(&fx.device, &q, shape.clone()))
                        .collect();
                    let mut chunk = TensorResumeChunk::default();
                    let start = std::time::Instant::now();
                    let mut outs = Vec::with_capacity(LAYERS);
                    for q in &qs {
                        outs.push(if tensor_route {
                            apply_tq_prefill_seq_major_resume_tensor_for_slot(
                                &fx.device,
                                &mut fx.registry,
                                &fx.cache.full_attn[0],
                                q,
                                seq_len,
                                cur_len,
                                cur_len + seq_len,
                                fx.capacity,
                                NH,
                                NKV,
                                HEAD_DIM,
                                SlotId(0),
                                &mut chunk,
                            )
                            .expect("tensor")
                        } else {
                            super::super::gpu_full_attn::apply_flash_attn_prefill_seq_major_resume_via_tq_cache_for_slot(
                                &fx.device,
                                &mut fx.registry,
                                &fx.cache.full_attn[0],
                                q,
                                seq_len,
                                cur_len,
                                cur_len + seq_len,
                                fx.capacity,
                                NH,
                                NKV,
                                HEAD_DIM,
                                SlotId(0),
                            )
                            .expect("bridge")
                        });
                    }
                    drain(&fx.device);
                    samples.push(start.elapsed().as_secs_f64() * 1e6 / LAYERS as f64);
                }
                samples.remove(0); // warm-up
                samples.sort_by(f64::total_cmp);
                samples
            };
            let tensor_us = time(&mut fx, true);
            let bridge_us = time(&mut fx, false);
            eprintln!(
                "[F3 bench] c={cur_len} n={seq_len} µs/layer median tensor={:.1} bridge={:.1} \
                 (tensor samples {tensor_us:.1?}, bridge samples {bridge_us:.1?})",
                tensor_us[SAMPLES / 2],
                bridge_us[SAMPLES / 2]
            );
        }
    }
}
