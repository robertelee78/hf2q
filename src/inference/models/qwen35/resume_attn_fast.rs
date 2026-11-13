//! Class A fast path of route (c): resumed Qwen prefill attention on TQ-only
//! full-attention slots, for chunks with a cached prefix (`cur_len > 0`) and
//! more than 32 queries.
//!
//! Route (c) ([`super::gpu_full_attn::apply_flash_attn_prefill_seq_major_resume_via_tq_cache_for_slot`])
//! allocates eight fresh buffers per FA layer (the F32 K/V dequant temps,
//! BF16 Q/K/V/out and the F32 result, about 60 MiB at `cur_len` 2048 and 132
//! MiB at 8192 for 512 queries). `MlxDevice::alloc_buffer` zero-fills every
//! byte on the CPU and registers each buffer with the residency set, and the
//! route ends in a host wait, so all of that is serial with the GPU.
//!
//! With `HF2Q_QWEN_RESUME_ATTN_FAST` on (the default), the route runs the
//! same dispatches in the same order on the same input bytes:
//! - the BF16 Q and output buffers come from the prefill's
//!   [`FaPrefillArena`], which resumed chunks allocate but did not use;
//! - the K/V temps, their BF16 casts and the F32 result are allocated once
//!   per prefill, by the first FA layer, in [`ResumeTqScratch`] on that arena;
//! - the dequant and the bridge share one command buffer (a memory barrier
//!   replaces the command-buffer boundary), committed without a host wait.
//!   The layer's ops6-7 consumes the result in queue order and keeps its own
//!   wait.
//!
//! Every element the attention reads is written earlier in the same layer,
//! so reused storage gives the bytes of fresh storage: outputs are bitwise
//! identical. The scratch lives until the forward's terminal wait, so no
//! buffer an in-flight command buffer references is dropped (the iter58b
//! residency-rescission contract of [`FaPrefillArena`]).
//!
//! Outside `forward_gpu_impl` (no arena: MTP drafts, decode, the source
//! teacher, direct helpers) route (c) is unchanged, as it is everywhere with
//! the switch off.

use std::sync::LazyLock;

use anyhow::{anyhow, Context, Result};
use mlx_native::{CommandEncoder, DType, KernelRegistry, MlxBuffer, MlxDevice};

use super::gpu_full_attn::{
    encode_flash_attn_prefill_seq_major_resume, validate_flash_attn_prefill_resume_shape,
    FaResumeBridgeBuffers,
};
use super::joint_step_options::parse_switch;
use super::kv_cache::FullAttnKvSlot;
use super::FaPrefillArena;
use crate::serve::multi_seq_kv::SlotId;

/// Kill switch, default on: unset, blank or `1`/`on`/`true` takes the fast
/// route; `0`/`off`/`false` keeps route (c) as it was.
pub(crate) const QWEN_RESUME_ATTN_FAST_ENV: &str = "HF2Q_QWEN_RESUME_ATTN_FAST";

/// Parse the switch; unset or blank is on, and an invalid value warns and
/// keeps the default (on).
pub(crate) fn parse_resume_attn_fast(raw: Option<&str>) -> bool {
    match raw.map(str::trim) {
        None | Some("") => true,
        Some(_) => parse_switch(QWEN_RESUME_ATTN_FAST_ENV, raw).unwrap_or_else(|error| {
            tracing::warn!(
                error,
                fallback = true,
                "invalid Qwen resumed-prefill attention switch"
            );
            true
        }),
    }
}

static RESUME_ATTN_FAST: LazyLock<bool> = LazyLock::new(|| {
    parse_resume_attn_fast(std::env::var(QWEN_RESUME_ATTN_FAST_ENV).ok().as_deref())
});

/// Whether `HF2Q_QWEN_RESUME_ATTN_FAST` is on (read once per process). Test
/// builds honour a [`ForceResumeAttnRoute`] scope on the calling thread first.
pub(crate) fn resume_attn_fast_enabled() -> bool {
    #[cfg(test)]
    if let Some(fast) = FORCED_ROUTE.with(std::cell::Cell::get) {
        return fast;
    }
    *RESUME_ATTN_FAST
}

#[cfg(test)]
thread_local! {
    static FORCED_ROUTE: std::cell::Cell<Option<bool>> = const { std::cell::Cell::new(None) };
    static FAST_ROUTE_CALLS: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };
}

/// Test-only: while alive, route (c) on this thread takes the fast path
/// (`fast`) or the shipped bridge (`!fast`) regardless of the switch, which
/// is read once per process.
#[cfg(test)]
pub(crate) struct ForceResumeAttnRoute {
    previous: Option<bool>,
}

#[cfg(test)]
impl ForceResumeAttnRoute {
    pub(crate) fn new(fast: bool) -> Self {
        Self {
            previous: FORCED_ROUTE.with(|cell| cell.replace(Some(fast))),
        }
    }
}

#[cfg(test)]
impl Drop for ForceResumeAttnRoute {
    fn drop(&mut self) {
        FORCED_ROUTE.with(|cell| cell.set(self.previous));
    }
}

/// Test-only: fast-route layer calls committed on this thread so far.
#[cfg(test)]
pub(crate) fn fast_route_calls() -> u64 {
    FAST_ROUTE_CALLS.with(std::cell::Cell::get)
}

/// Route (c) buffers that depend on the attended length, plus the F32
/// result, for one prefill. Same dtypes and shapes as the buffers route (c)
/// allocates per layer.
pub(crate) struct ResumeTqScratch {
    seq_len: u32,
    kv_seq_len: u32,
    /// F32 `[n_kv_heads, kv_seq_len, head_dim]`: unrotated TQ dequant of K.
    temp_k: MlxBuffer,
    /// Same for V.
    temp_v: MlxBuffer,
    /// BF16 `[1, n_kv_heads, kv_seq_len, head_dim]`: cast of `temp_k`.
    k_bf16: MlxBuffer,
    /// Same for V.
    v_bf16: MlxBuffer,
    /// F32 `[seq_len, n_heads, head_dim]`: the route's result. Each FA layer
    /// gets a clone; its ops6-7 consumes it before the next FA layer
    /// rewrites it (queue order, plus that layer's host wait).
    out_seq: MlxBuffer,
}

impl ResumeTqScratch {
    fn new(
        device: &MlxDevice,
        seq_len: u32,
        kv_seq_len: u32,
        n_heads: u32,
        n_kv_heads: u32,
        head_dim: u32,
    ) -> Result<Self> {
        let (seq, kv) = (seq_len as usize, kv_seq_len as usize);
        let (nh, nkv, d) = (n_heads as usize, n_kv_heads as usize, head_dim as usize);
        let kv_elems = nkv * kv * d;
        let alloc = |bytes: usize, dtype: DType, shape: Vec<usize>, label: &str| {
            device
                .alloc_buffer(bytes, dtype, shape)
                .map_err(|e| anyhow!("ResumeTqScratch alloc {label}: {e}"))
        };
        Ok(Self {
            seq_len,
            kv_seq_len,
            temp_k: alloc(kv_elems * 4, DType::F32, vec![nkv, kv, d], "temp_k")?,
            temp_v: alloc(kv_elems * 4, DType::F32, vec![nkv, kv, d], "temp_v")?,
            k_bf16: alloc(kv_elems * 2, DType::BF16, vec![1, nkv, kv, d], "k_bf16")?,
            v_bf16: alloc(kv_elems * 2, DType::BF16, vec![1, nkv, kv, d], "v_bf16")?,
            out_seq: alloc(seq * nh * d * 4, DType::F32, vec![seq, nh, d], "out_seq")?,
        })
    }
}

/// The arena's buffers for one resumed layer, allocating the
/// [`ResumeTqScratch`] on the prefill's first resumed FA layer. Every FA
/// layer of one prefill attends the same `(seq_len, kv_seq_len)`; a
/// different shape is a caller bug and errors.
fn arena_buffers<'a>(
    arena: &'a mut FaPrefillArena,
    device: &MlxDevice,
    seq_len: u32,
    kv_seq_len: u32,
) -> Result<(FaResumeBridgeBuffers<'a>, &'a MlxBuffer, &'a MlxBuffer)> {
    match &arena.resume_tq {
        Some(scratch) if scratch.seq_len != seq_len || scratch.kv_seq_len != kv_seq_len => {
            return Err(anyhow!(
                "resume scratch was allocated for seq_len {} / kv_seq_len {} in this \
                 prefill, but a layer attends seq_len {seq_len} / kv_seq_len {kv_seq_len}",
                scratch.seq_len,
                scratch.kv_seq_len
            ));
        }
        Some(_) => {}
        None => {
            arena.resume_tq = Some(ResumeTqScratch::new(
                device,
                seq_len,
                kv_seq_len,
                arena.n_heads,
                arena.n_kv_heads,
                arena.head_dim,
            )?);
        }
    }
    let arena: &FaPrefillArena = arena;
    let scratch = arena
        .resume_tq
        .as_ref()
        .ok_or_else(|| anyhow!("resume scratch missing after allocation"))?;
    Ok((
        FaResumeBridgeBuffers {
            q_bf16_seq: &arena.q_bf16_seq,
            q_bf16_hm: &arena.q_bf16_hm,
            k_bf16: &scratch.k_bf16,
            v_bf16: &scratch.v_bf16,
            out_bf16_hm: &arena.out_bf16_hm,
            out_seq: &scratch.out_seq,
        },
        &scratch.temp_k,
        &scratch.temp_v,
    ))
}

/// Encode route (c) for one FA layer into `enc` on the arena's scratch,
/// without committing: the unrotated TQ dequant of K and V over `[0, L)`,
/// a barrier, then the BF16 bridge. Returns (a clone of) the F32
/// `[seq_len, n_heads, head_dim]` result.
#[allow(clippy::too_many_arguments)]
pub(crate) fn encode_resume_via_tq_cache_fast(
    enc: &mut CommandEncoder,
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
    arena: &mut FaPrefillArena,
) -> Result<MlxBuffer> {
    if slot.tq.is_none() {
        return Err(anyhow!(
            "apply_flash_attn_prefill_seq_major_resume_via_tq_cache_fast: slot.tq is None — \
             slot was not constructed in TQ-active mode. Caller routing bug."
        ));
    }
    // Route (c) attends tight temps: kv_capacity = kv_seq_len.
    validate_flash_attn_prefill_resume_shape(seq_len, cur_len, kv_seq_len, kv_seq_len, head_dim)?;
    arena.validate_fits(seq_len, n_heads, n_kv_heads, head_dim)?;
    let (bufs, temp_k, temp_v) = arena_buffers(arena, device, seq_len, kv_seq_len)?;

    for (is_k, temp) in [(true, temp_k), (false, temp_v)] {
        slot.dequant_seq_into_f32_unrotated_for_slot(
            is_k,
            kv_seq_len,
            /*start_pos=*/ 0,
            cache_capacity,
            n_kv_heads,
            head_dim,
            slot_id,
            temp,
            enc,
            registry,
            device,
        )
        .with_context(|| {
            format!(
                "dequant {} seq → scratch F32 unrotated",
                if is_k { "K" } else { "V" }
            )
        })?;
    }
    // Replaces route (c)'s command-buffer boundary: the bridge's K/V casts
    // read what the undo just wrote.
    enc.memory_barrier();
    encode_flash_attn_prefill_seq_major_resume(
        enc,
        device,
        registry,
        q_seq_major,
        temp_k,
        temp_v,
        &bufs,
        seq_len,
        cur_len,
        kv_seq_len,
        /*kv_capacity=*/ kv_seq_len,
        n_heads,
        n_kv_heads,
        head_dim,
    )
    .context("apply_flash_attn_prefill_seq_major_resume (TQ-decoded fast path)")?;
    Ok(bufs.out_seq.clone())
}

/// Route (c) with `HF2Q_QWEN_RESUME_ATTN_FAST` on: [`encode_resume_via_tq_cache_fast`]
/// in one command buffer, committed without a host wait.
#[allow(clippy::too_many_arguments)]
pub(crate) fn apply_flash_attn_prefill_seq_major_resume_via_tq_cache_fast(
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
    arena: &mut FaPrefillArena,
) -> Result<MlxBuffer> {
    let mut enc = device
        .command_encoder()
        .map(crate::inference::native_observation::encoder)
        .context("resume attention fast path encoder")?;
    let out = encode_resume_via_tq_cache_fast(
        &mut enc,
        device,
        registry,
        slot,
        q_seq_major,
        seq_len,
        cur_len,
        kv_seq_len,
        cache_capacity,
        n_heads,
        n_kv_heads,
        head_dim,
        slot_id,
        arena,
    )?;
    #[cfg(test)]
    FAST_ROUTE_CALLS.with(|calls| calls.set(calls.get() + 1));
    enc.commit_labeled("layer.full_attn.tq_bridge_resume");
    // Hand the submitted encoder over: a pipelined forward keeps it for
    // its next host wait; otherwise it is released here, so the native
    // observer retains this command before the layer's next wait checks
    // the queue.
    super::encoder_stage::retain_submitted(enc, "layer.full_attn.tq_bridge_resume");
    Ok(out)
}

#[cfg(test)]
mod tests {
    use std::time::Instant;

    use mlx_native::ops::elementwise::{cast, CastDirection};
    use mlx_native::ops::flash_attn_prefill::{
        dispatch_flash_attn_prefill_bf16_d256_resume, FlashAttnPrefillResumeParams,
    };
    use mlx_native::ops::fwht_standalone::dispatch_fwht_sign_undo_f32;
    use mlx_native::ops::transpose::{permute_021_bf16, permute_021_bf16_to_f32, permute_021_f32};
    use rand::{rngs::StdRng, Rng, SeedableRng};

    use super::*;
    use crate::inference::models::qwen35::gpu_full_attn::apply_flash_attn_prefill_seq_major_resume_via_tq_cache_for_slot;
    use crate::inference::models::qwen35::kv_cache::HybridKvCache;
    use crate::inference::models::qwen35::{default_layer_types, Qwen35Config, Qwen35Variant};

    const NH: u32 = 24;
    const NKV: u32 = 4;
    const HEAD_DIM: u32 = 256;
    const D: usize = HEAD_DIM as usize;
    /// Slot capacity past the attended length, so the dequant reads the
    /// cache at a stride other than `L`, as in production.
    const CAPACITY_SLACK: u32 = 64;

    /// Qwen3.8 full-attention shapes; every layer is full attention so a
    /// cache holds one TQ slot per layer and no linear-attention state.
    fn qwen38_attn_cfg(layers: u32) -> Qwen35Config {
        Qwen35Config {
            variant: Qwen35Variant::Dense,
            hidden_size: 5120,
            num_hidden_layers: layers,
            num_attention_heads: NH,
            num_key_value_heads: NKV,
            head_dim: HEAD_DIM,
            linear_num_key_heads: 16,
            linear_num_value_heads: 48,
            linear_key_head_dim: 128,
            linear_value_head_dim: 128,
            linear_conv_kernel_dim: 4,
            full_attention_interval: 1,
            layer_types: default_layer_types(layers, 1),
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

    fn read(buffer: &MlxBuffer) -> Vec<f32> {
        buffer.as_slice::<f32>().expect("read back").to_vec()
    }

    fn assert_bitwise(label: &str, got: &[f32], want: &[f32]) {
        assert_eq!(got.len(), want.len(), "{label}: length");
        assert!(
            want.iter().any(|x| *x != 0.0),
            "{label}: reference is all zero"
        );
        let differ = got
            .iter()
            .zip(want)
            .filter(|(a, b)| a.to_bits() != b.to_bits())
            .count();
        assert_eq!(
            differ,
            0,
            "{label}: {differ} of {} elements differ",
            got.len()
        );
    }

    /// `layers` TQ-only slots, each holding `total` positions of random K/V
    /// written by the production `hadamard_quantize` encoder, and per layer
    /// `queries` rows of random Q.
    struct Fixture {
        device: MlxDevice,
        registry: KernelRegistry,
        cache: HybridKvCache,
        capacity: u32,
        /// Per layer `[queries, NH, D]`.
        q: Vec<Vec<f32>>,
    }

    impl Fixture {
        fn new(total: u32, queries: u32, layers: u32, seed: u64) -> Option<Self> {
            let device = match MlxDevice::new() {
                Ok(device) => device,
                Err(e) => {
                    eprintln!("skipping: no Metal device: {e}");
                    return None;
                }
            };
            let mut registry = KernelRegistry::new();
            mlx_native::ops::flash_attn_prefill::register(&mut registry);
            let capacity = total + CAPACITY_SLACK;
            let mut cache = HybridKvCache::new_with_options(
                &qwen38_attn_cfg(layers),
                &device,
                capacity,
                1,
                true,
            )
            .expect("TQ cache");
            let mut rng = StdRng::seed_from_u64(seed);
            let kv_elems = total as usize * NKV as usize * D;
            for layer in 0..layers as usize {
                let k: Vec<f32> = (0..kv_elems).map(|_| rng.gen_range(-1.0f32..1.0)).collect();
                let v: Vec<f32> = (0..kv_elems).map(|_| rng.gen_range(-1.0f32..1.0)).collect();
                let shape = vec![total as usize, NKV as usize, D];
                let (k_gpu, v_gpu) = (
                    upload(&device, &k, shape.clone()),
                    upload(&device, &v, shape),
                );
                let slot = &mut cache.full_attn[layer];
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
            }
            let q = (0..layers)
                .map(|_| {
                    (0..queries as usize * NH as usize * D)
                        .map(|_| rng.gen_range(-2.0f32..2.0))
                        .collect()
                })
                .collect();
            Some(Self {
                device,
                registry,
                cache,
                capacity,
                q,
            })
        }

        fn upload_q(&self, layer: usize, rows: u32) -> MlxBuffer {
            let row = NH as usize * D;
            upload(
                &self.device,
                &self.q[layer][..rows as usize * row],
                vec![rows as usize, NH as usize, D],
            )
        }

        /// Route (c) as shipped (ends in its own host wait).
        fn bridge_buf(
            &mut self,
            layer: usize,
            q: &MlxBuffer,
            cur_len: u32,
            rows: u32,
        ) -> MlxBuffer {
            apply_flash_attn_prefill_seq_major_resume_via_tq_cache_for_slot(
                &self.device,
                &mut self.registry,
                &self.cache.full_attn[layer],
                q,
                rows,
                cur_len,
                cur_len + rows,
                self.capacity,
                NH,
                NKV,
                HEAD_DIM,
                SlotId(0),
            )
            .expect("bridge")
        }

        fn bridge(&mut self, layer: usize, cur_len: u32, rows: u32) -> Vec<f32> {
            let q = self.upload_q(layer, rows);
            read(&self.bridge_buf(layer, &q, cur_len, rows))
        }

        /// The fast route, committed without a wait; `q` must outlive the
        /// caller's drain.
        fn fast(
            &mut self,
            layer: usize,
            q: &MlxBuffer,
            cur_len: u32,
            rows: u32,
            arena: &mut FaPrefillArena,
        ) -> Result<MlxBuffer> {
            apply_flash_attn_prefill_seq_major_resume_via_tq_cache_fast(
                &self.device,
                &mut self.registry,
                &self.cache.full_attn[layer],
                q,
                rows,
                cur_len,
                cur_len + rows,
                self.capacity,
                NH,
                NKV,
                HEAD_DIM,
                SlotId(0),
                arena,
            )
        }

        fn arena(&self, rows: u32) -> FaPrefillArena {
            FaPrefillArena::new(&self.device, rows, NH, NKV, HEAD_DIM).expect("FaPrefillArena")
        }
    }

    #[test]
    fn resume_attn_fast_switch_parses_and_defaults_on() {
        assert!(parse_resume_attn_fast(None));
        assert!(parse_resume_attn_fast(Some("  ")));
        for on in ["1", "on", "TRUE"] {
            assert!(parse_resume_attn_fast(Some(on)), "{on}");
        }
        for off in ["0", "off", "False", " 0 "] {
            assert!(!parse_resume_attn_fast(Some(off)), "{off}");
        }
        // Invalid: warns and keeps the default (on).
        assert!(parse_resume_attn_fast(Some("fast")));
    }

    /// Bitwise against route (c) at Qwen3.8 shapes, including attended
    /// lengths that are not multiples of 64 (or of the kernel's 16-row K
    /// tile) and a query count that is not a multiple of 32. The second pass
    /// reuses the arena's scratch.
    #[test]
    fn resume_attn_fast_matches_bridge_bitwise_at_qwen38_shapes() {
        let _gpu = crate::inference::hf2q_gpu_test_lock();
        let cases = [
            (64u32, 64u32),
            (64, 512),
            (2048, 64),
            (2048, 512),
            (100, 77),
            (2003, 512),
        ];
        for (seed, (cur_len, rows)) in cases.into_iter().enumerate() {
            let Some(mut fx) = Fixture::new(cur_len + rows, rows, 1, 11 + seed as u64) else {
                return;
            };
            let bridge = fx.bridge(0, cur_len, rows);
            let mut arena = fx.arena(rows);
            for pass in 0..2 {
                let q = fx.upload_q(0, rows);
                let out = fx.fast(0, &q, cur_len, rows, &mut arena).expect("fast");
                drain(&fx.device);
                assert_bitwise(
                    &format!("c={cur_len} n={rows} pass {pass}"),
                    &read(&out),
                    &bridge,
                );
            }
        }
    }

    /// One chunk's 16 FA layers through one arena, with no host wait between
    /// layers: each layer's result is consumed by a queued GPU copy (the
    /// role of ops6-7) before the next layer rewrites the shared scratch.
    #[test]
    fn resume_attn_fast_sixteen_layer_chunk_matches_bridge_bitwise() {
        let _gpu = crate::inference::hf2q_gpu_test_lock();
        const LAYERS: u32 = 16;
        let (cur_len, rows) = (2048u32, 512u32);
        let Some(mut fx) = Fixture::new(cur_len + rows, rows, LAYERS, 23) else {
            return;
        };
        let bridge: Vec<Vec<f32>> = (0..LAYERS as usize)
            .map(|layer| fx.bridge(layer, cur_len, rows))
            .collect();
        let qs: Vec<MlxBuffer> = (0..LAYERS as usize).map(|l| fx.upload_q(l, rows)).collect();
        let out_elems = rows as usize * NH as usize * D;
        let copies: Vec<MlxBuffer> = (0..LAYERS)
            .map(|_| {
                fx.device
                    .alloc_buffer(
                        out_elems * 4,
                        DType::F32,
                        vec![rows as usize, NH as usize, D],
                    )
                    .expect("alloc copy")
            })
            .collect();
        let mut arena = fx.arena(rows);
        for layer in 0..LAYERS as usize {
            let out = fx
                .fast(layer, &qs[layer], cur_len, rows, &mut arena)
                .expect("fast");
            let mut enc = fx.device.command_encoder().expect("consumer encoder");
            // [1, rows·NH, D] → [rows·NH, 1, D]: a byte copy.
            permute_021_f32(
                &mut enc,
                &mut fx.registry,
                fx.device.metal_device(),
                &out,
                &copies[layer],
                1,
                rows as usize * NH as usize,
                D,
            )
            .expect("consumer copy");
            enc.commit();
        }
        drain(&fx.device);
        for layer in 0..LAYERS as usize {
            assert_bitwise(
                &format!("layer {layer}"),
                &read(&copies[layer]),
                &bridge[layer],
            );
        }
    }

    #[test]
    fn resume_attn_fast_rejects_a_shape_change_within_a_prefill() {
        let _gpu = crate::inference::hf2q_gpu_test_lock();
        let rows = 64u32;
        let Some(mut fx) = Fixture::new(3 * rows, rows, 1, 31) else {
            return;
        };
        let q = fx.upload_q(0, rows);
        let mut arena = fx.arena(rows);
        fx.fast(0, &q, rows, rows, &mut arena).expect("first layer");
        let error = fx
            .fast(0, &q, 2 * rows, rows, &mut arena)
            .expect_err("a different kv_seq_len in one prefill must fail");
        drain(&fx.device);
        assert!(format!("{error:#}").contains("resume scratch"), "{error:#}");
    }

    fn median(mut values: Vec<f64>) -> f64 {
        values.sort_by(f64::total_cmp);
        values[values.len() / 2]
    }

    fn elapsed_us(start: Instant) -> f64 {
        start.elapsed().as_secs_f64() * 1e6
    }

    fn gpu_us(enc: &mut CommandEncoder) -> f64 {
        let (start, end) = enc.commit_wait_with_gpu_time().expect("gpu time");
        (end - start) * 1e6
    }

    /// Per-layer wall of one 16-layer chunk, route (c) against the fast
    /// route. `fast+wait` drains after every layer, standing in for the
    /// ops6-7 host wait every resumed FA layer still has in production;
    /// `fast` drains once per chunk. Each fast sample opens a fresh arena,
    /// so the first layer's scratch allocation is in the sample.
    /// `cargo test --release --bin hf2q resume_attn_fast_bench -- --ignored --nocapture`
    #[test]
    #[ignore = "timing micro-bench; run explicitly"]
    fn resume_attn_fast_bench() {
        let _gpu = crate::inference::hf2q_gpu_test_lock();
        const LAYERS: usize = 16;
        const SAMPLES: usize = 5;
        let rows = 512u32;
        for cur_len in [2048u32, 8192] {
            let Some(mut fx) = Fixture::new(cur_len + rows, rows, 1, 5) else {
                return;
            };
            let mut results = Vec::new();
            for mode in ["bridge", "fast+wait", "fast"] {
                let mut samples = Vec::new();
                for _ in 0..=SAMPLES {
                    let qs: Vec<MlxBuffer> = (0..LAYERS).map(|_| fx.upload_q(0, rows)).collect();
                    let mut arena = fx.arena(rows);
                    let mut outs = Vec::with_capacity(LAYERS);
                    let start = Instant::now();
                    for q in &qs {
                        outs.push(match mode {
                            "bridge" => fx.bridge_buf(0, q, cur_len, rows),
                            _ => fx.fast(0, q, cur_len, rows, &mut arena).expect("fast"),
                        });
                        if mode == "fast+wait" {
                            drain(&fx.device);
                        }
                    }
                    drain(&fx.device);
                    samples.push(elapsed_us(start) / LAYERS as f64);
                }
                samples.remove(0); // warm-up
                results.push((mode, median(samples.clone()), samples));
            }
            for (mode, med, samples) in results {
                eprintln!(
                    "[resume-attn bench] c={cur_len} n={rows} µs/layer median {mode}={med:.1} \
                     (samples {samples:.1?})"
                );
            }
        }
    }

    /// Stage breakdown of route (c) per layer: CPU wall of each stage of the
    /// shipped two-command-buffer structure (replicated with the same calls,
    /// checked bitwise against the shipped route), GPU time per command
    /// buffer and per op, and the allocation cost per buffer. Prints
    /// markdown.
    /// `cargo test --release --bin hf2q resume_attn_profile -- --ignored --nocapture`
    #[test]
    #[ignore = "timing profile; run explicitly"]
    fn resume_attn_profile() {
        let _gpu = crate::inference::hf2q_gpu_test_lock();
        const LAYERS: usize = 16;
        const SAMPLES: usize = 3;
        let rows = 512u32;
        for cur_len in [2048u32, 8192] {
            let kv = cur_len + rows;
            let Some(mut fx) = Fixture::new(kv, rows, 1, 5) else {
                return;
            };
            let device = fx.device.clone();
            let (seq, nh, nkv) = (rows as usize, NH as usize, NKV as usize);
            let kv_elems = nkv * kv as usize * D;
            let q_elems = seq * nh * D;
            let sizes: [(&str, usize, DType, Vec<usize>); 8] = [
                (
                    "temp_k F32",
                    kv_elems * 4,
                    DType::F32,
                    vec![nkv, kv as usize, D],
                ),
                (
                    "temp_v F32",
                    kv_elems * 4,
                    DType::F32,
                    vec![nkv, kv as usize, D],
                ),
                ("q_bf16_seq", q_elems * 2, DType::BF16, vec![seq, nh, D]),
                ("q_bf16_hm", q_elems * 2, DType::BF16, vec![1, nh, seq, D]),
                (
                    "k_bf16",
                    kv_elems * 2,
                    DType::BF16,
                    vec![1, nkv, kv as usize, D],
                ),
                (
                    "v_bf16",
                    kv_elems * 2,
                    DType::BF16,
                    vec![1, nkv, kv as usize, D],
                ),
                ("out_bf16_hm", q_elems * 2, DType::BF16, vec![1, nh, seq, D]),
                ("out_seq F32", q_elems * 4, DType::F32, vec![seq, nh, D]),
            ];
            let alloc = |i: usize| -> MlxBuffer {
                let (_, bytes, dtype, shape) = &sizes[i];
                device
                    .alloc_buffer(*bytes, *dtype, shape.clone())
                    .expect("alloc")
            };
            let q = fx.upload_q(0, rows);
            let slot_dequant =
                |fx: &mut Fixture, enc: &mut CommandEncoder, is_k: bool, dst: &MlxBuffer| {
                    fx.cache.full_attn[0]
                        .dequant_seq_into_f32_unrotated_for_slot(
                            is_k,
                            kv,
                            0,
                            fx.capacity,
                            NKV,
                            HEAD_DIM,
                            SlotId(0),
                            dst,
                            enc,
                            &mut fx.registry,
                            &device,
                        )
                        .expect("dequant");
                };
            let encode_bridge = |fx: &mut Fixture, enc: &mut CommandEncoder, b: &[MlxBuffer]| {
                encode_flash_attn_prefill_seq_major_resume(
                    enc,
                    &device,
                    &mut fx.registry,
                    &q,
                    &b[0],
                    &b[1],
                    &FaResumeBridgeBuffers {
                        q_bf16_seq: &b[2],
                        q_bf16_hm: &b[3],
                        k_bf16: &b[4],
                        v_bf16: &b[5],
                        out_bf16_hm: &b[6],
                        out_seq: &b[7],
                    },
                    rows,
                    cur_len,
                    kv,
                    kv,
                    NH,
                    NKV,
                    HEAD_DIM,
                )
                .expect("bridge encode");
            };

            // ── A. CPU wall per stage, shipped structure ──
            let stages = [
                "alloc temp K/V (2, zero-filled)",
                "encode dequant + undo",
                "commit dequant CB",
                "alloc bridge (6, zero-filled)",
                "encode bridge",
                "commit_and_wait bridge CB",
                "drop 8 buffers",
                "replica total",
                "route (c) as shipped",
            ];
            let mut wall: Vec<Vec<f64>> = vec![Vec::new(); stages.len()];
            let mut replica_out = None;
            for sample in 0..=SAMPLES {
                for _ in 0..LAYERS {
                    let total = Instant::now();
                    let t = Instant::now();
                    let mut b: Vec<MlxBuffer> = (0..2).map(alloc).collect();
                    let w0 = elapsed_us(t);
                    let t = Instant::now();
                    let mut enc = device
                        .command_encoder()
                        .map(crate::inference::native_observation::encoder)
                        .expect("encoder");
                    slot_dequant(&mut fx, &mut enc, true, &b[0]);
                    slot_dequant(&mut fx, &mut enc, false, &b[1]);
                    let w1 = elapsed_us(t);
                    let t = Instant::now();
                    enc.commit_labeled("layer.full_attn.tq_bridge_dequant");
                    drop(enc);
                    let w2 = elapsed_us(t);
                    let t = Instant::now();
                    b.extend((2..8).map(alloc));
                    let w3 = elapsed_us(t);
                    let t = Instant::now();
                    let mut enc = device
                        .command_encoder()
                        .map(crate::inference::native_observation::encoder)
                        .expect("encoder");
                    encode_bridge(&mut fx, &mut enc, &b);
                    let w4 = elapsed_us(t);
                    let t = Instant::now();
                    enc.commit_and_wait().expect("bridge wait");
                    let w5 = elapsed_us(t);
                    if replica_out.is_none() {
                        replica_out = Some(read(&b[7]));
                    }
                    let t = Instant::now();
                    drop(b);
                    let w6 = elapsed_us(t);
                    let w7 = elapsed_us(total);
                    let t = Instant::now();
                    drop(fx.bridge_buf(0, &q, cur_len, rows));
                    let w8 = elapsed_us(t);
                    if sample > 0 {
                        for (i, w) in [w0, w1, w2, w3, w4, w5, w6, w7, w8].into_iter().enumerate() {
                            wall[i].push(w);
                        }
                    }
                }
            }
            let shipped = read(&fx.bridge_buf(0, &q, cur_len, rows));
            assert_bitwise(
                "replica vs route (c)",
                &replica_out.expect("replica ran"),
                &shipped,
            );

            // Pre-allocated buffers for the GPU-only measurements.
            let b: Vec<MlxBuffer> = (0..8).map(alloc).collect();

            // ── B. GPU time per command buffer ──
            let mut cb_dequant = Vec::new();
            let mut cb_bridge = Vec::new();
            let mut cb_fast = Vec::new();
            let mut fast_encode = Vec::new();
            let mut arena = fx.arena(rows);
            for sample in 0..=SAMPLES * LAYERS {
                let mut enc = device.command_encoder().expect("encoder");
                slot_dequant(&mut fx, &mut enc, true, &b[0]);
                slot_dequant(&mut fx, &mut enc, false, &b[1]);
                let g0 = gpu_us(&mut enc);
                let mut enc = device.command_encoder().expect("encoder");
                encode_bridge(&mut fx, &mut enc, &b);
                let g1 = gpu_us(&mut enc);
                let t = Instant::now();
                let mut enc = device.command_encoder().expect("encoder");
                encode_resume_via_tq_cache_fast(
                    &mut enc,
                    &device,
                    &mut fx.registry,
                    &fx.cache.full_attn[0],
                    &q,
                    rows,
                    cur_len,
                    kv,
                    fx.capacity,
                    NH,
                    NKV,
                    HEAD_DIM,
                    SlotId(0),
                    &mut arena,
                )
                .expect("fast encode");
                let e = elapsed_us(t);
                let g2 = gpu_us(&mut enc);
                if sample > 0 {
                    cb_dequant.push(g0);
                    cb_bridge.push(g1);
                    cb_fast.push(g2);
                    fast_encode.push(e);
                }
            }

            // ── C. GPU time per op (one op per command buffer) ──
            let tight = kv * NKV;
            let ops = [
                "dequant K (rotated)",
                "FWHT undo K",
                "dequant V (rotated)",
                "FWHT undo V",
                "cast Q F32→BF16",
                "permute Q",
                "cast K F32→BF16",
                "cast V F32→BF16",
                "flash_attn_prefill_bf16_d256 (resume)",
                "permute out BF16→F32",
            ];
            let mut per_op: Vec<Vec<f64>> = vec![Vec::new(); ops.len()];
            for sample in 0..=SAMPLES * LAYERS {
                let mut times = Vec::with_capacity(ops.len());
                for op in 0..ops.len() {
                    let mut enc = device.command_encoder().expect("encoder");
                    let r = &mut fx.registry;
                    let md = device.metal_device();
                    match op {
                        0 | 2 => fx.cache.full_attn[0]
                            .dequant_seq_into_f32_for_slot(
                                op == 0,
                                kv,
                                0,
                                fx.capacity,
                                NKV,
                                HEAD_DIM,
                                SlotId(0),
                                &b[op / 2],
                                &mut enc,
                                r,
                                &device,
                            )
                            .expect("dequant"),
                        1 | 3 => dispatch_fwht_sign_undo_f32(
                            &mut enc,
                            r,
                            md,
                            &b[op / 2],
                            tight,
                            HEAD_DIM,
                        )
                        .expect("undo"),
                        4 => cast(
                            &mut enc,
                            r,
                            md,
                            &q,
                            &b[2],
                            q_elems,
                            CastDirection::F32ToBF16,
                        )
                        .expect("cast q"),
                        5 => permute_021_bf16(&mut enc, r, md, &b[2], &b[3], seq, nh, D)
                            .expect("perm q"),
                        6 | 7 => cast(
                            &mut enc,
                            r,
                            md,
                            &b[op - 6],
                            &b[op - 2],
                            kv_elems,
                            CastDirection::F32ToBF16,
                        )
                        .expect("cast kv"),
                        8 => dispatch_flash_attn_prefill_bf16_d256_resume(
                            &mut enc,
                            &device,
                            r,
                            &b[3],
                            &b[4],
                            &b[5],
                            &b[6],
                            &FlashAttnPrefillResumeParams {
                                n_heads: NH,
                                n_kv_heads: NKV,
                                head_dim: HEAD_DIM,
                                seq_len_q: rows,
                                seq_len_k: kv,
                                batch: 1,
                                scale: 1.0 / 16.0,
                                do_causal: true,
                                q_offset_in_k: cur_len,
                                kv_capacity: kv,
                            },
                        )
                        .expect("fa"),
                        _ => permute_021_bf16_to_f32(&mut enc, r, md, &b[6], &b[7], nh, seq, D)
                            .expect("perm out"),
                    }
                    times.push(gpu_us(&mut enc));
                }
                if sample > 0 {
                    for (i, t) in times.into_iter().enumerate() {
                        per_op[i].push(t);
                    }
                }
            }

            // ── D. Allocation cost per buffer ──
            let mut alloc_rows = Vec::new();
            for (i, (label, bytes, dtype, shape)) in sizes.iter().enumerate() {
                let (mut zeroed, mut dropped, mut raw, mut raw_dropped) =
                    (Vec::new(), Vec::new(), Vec::new(), Vec::new());
                for _ in 0..LAYERS {
                    let t = Instant::now();
                    let buffer = alloc(i);
                    zeroed.push(elapsed_us(t));
                    let t = Instant::now();
                    drop(buffer);
                    dropped.push(elapsed_us(t));
                    let t = Instant::now();
                    // SAFETY: never bound to a dispatch nor read; dropped at once.
                    let buffer =
                        unsafe { device.alloc_buffer_for_overwrite(*bytes, *dtype, shape.clone()) }
                            .expect("alloc for overwrite");
                    raw.push(elapsed_us(t));
                    let t = Instant::now();
                    drop(buffer);
                    raw_dropped.push(elapsed_us(t));
                }
                alloc_rows.push(format!(
                    "| {label} | {:.2} | {:.1} | {:.1} | {:.1} | {:.1} |",
                    *bytes as f64 / (1 << 20) as f64,
                    median(zeroed),
                    median(dropped),
                    median(raw),
                    median(raw_dropped)
                ));
            }

            let mut md = format!(
                "\n### c = {cur_len}, n = {rows} (L = {kv}), µs per layer, medians over {} layers\n\n\
                 #### A. CPU wall per stage (shipped structure)\n\n| stage | µs |\n|---|---:|\n",
                SAMPLES * LAYERS
            );
            for (stage, w) in stages.iter().zip(wall) {
                md += &format!("| {stage} | {:.1} |\n", median(w));
            }
            md += "\n#### B. GPU time per command buffer\n\n| command buffer | GPU µs |\n|---|---:|\n";
            md += &format!(
                "| dequant (K, V: dequant + undo) | {:.1} |\n",
                median(cb_dequant)
            );
            md += &format!(
                "| bridge (casts, permutes, kernel) | {:.1} |\n",
                median(cb_bridge)
            );
            md += &format!(
                "| fast route, one CB (same dispatches) | {:.1} |\n",
                median(cb_fast)
            );
            md += &format!(
                "| fast route CPU encode (arena warm) | {:.1} |\n",
                median(fast_encode)
            );
            md += "\n#### C. GPU time per op (one op per CB)\n\n| op | GPU µs |\n|---|---:|\n";
            for (op, t) in ops.iter().zip(per_op) {
                md += &format!("| {op} | {:.1} |\n", median(t));
            }
            md += "\n#### D. Allocation per buffer (CPU µs)\n\n\
                   | buffer | MiB | alloc_buffer (zero-fill) | drop | alloc_for_overwrite | drop |\n\
                   |---|---:|---:|---:|---:|---:|\n";
            for row in alloc_rows {
                md += &row;
                md += "\n";
            }
            eprintln!("{md}");
        }
    }
}
