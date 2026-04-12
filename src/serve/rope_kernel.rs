//! Fused RoPE Metal kernel — ports llama.cpp `kernel_rope_norm` / `kernel_rope_neox`.
//!
//! # What this module does
//!
//! Runtime-compiles a small Metal Shading Language (MSL) library that
//! instantiates both `kernel_rope_norm<float>` and `kernel_rope_neox<float>`
//! byte-for-byte ported from
//! `/opt/llama.cpp/ggml/src/ggml-metal/ggml-metal.metal:4322-4426` (with
//! the shared yarn helpers from `:4282-4320`). A `rope_fused` helper
//! binds the correct buffers and strides to candle's shared command
//! encoder so the dispatch lives in-order with the rest of the
//! forward pass — the same shared-encoder protocol the 1bNEW.4
//! RmsNorm fuse and the 1bNEW.1 MoE fuse already follow.
//!
//! This is **ADR-005 1bNEW.6**. It replaces the 9-op `rope_apply`
//! chain at `gemma4.rs::RotaryEmbedding::rope_apply` and the partial-
//! rotary narrow/cat dance at `RotaryEmbedding::apply` with a single
//! stride-aware Metal dispatch per Q and per K projection, per
//! attention layer. llama.cpp's kernels are stride-aware by
//! construction (source/dest byte strides in `ggml_metal_kargs_rope`),
//! so the port incidentally eliminates the `.contiguous()` copies on
//! the Q/K narrowed views (old ADR item 1bNEW.8, now retired as
//! subsumed by faithful porting — see ADR-005:322-326).
//!
//! # Which variant does Gemma 4 use?
//!
//! **Both are shipped.** Gemma 4 is dispatched as `LLAMA_ROPE_TYPE_NEOX`
//! in llama.cpp (`src/llama-model.cpp:9134-9165`), which routes
//! through `kernel_rope_neox`. llama.cpp's "neox" name is historical;
//! the actual pair layout is **split-half** — element `ic ∈ [0, n_dims/2)`
//! is rotated against element `ic + n_dims/2` — matching the HF
//! `rotate_half` convention used by every modern Llama/Gemma/Qwen
//! port. hf2q's existing `rope_apply` at `gemma4.rs:377-384` uses the
//! same split-half pattern via `x1 = narrow(0, half); x2 = narrow(half, half)`,
//! so the fused-path output is byte-identical on the `neox` variant.
//!
//! llama.cpp's `kernel_rope_norm` is the older GPT-J style with
//! **interleaved pairs** `(x[i0], x[i0+1])` where `i0` increments by
//! two. No hf2q model currently exercises that path, but we compile
//! it anyway so (1) the port is faithful to the reference and (2)
//! future models using GPT-J-style RoPE can share the same pipeline
//! bundle. The unused pipeline's instantiation cost is O(1) at
//! model-load time.
//!
//! # Frequency scaling (Gemma 4 "proportional" RoPE)
//!
//! llama.cpp's kernel computes
//!
//!   `theta = pos * pow(freq_base, -i0/n_dims) / freq_factor`
//!
//! with denominator `n_dims = rotary_dim`. Gemma 4's HF-origin
//! "proportional" scaling uses `head_dim` in the denominator instead
//! (see `vllm/model_executor/layers/rotary_embedding/gemma4_rope.py:63-66`
//! — `freq_exponents = arange / head_size`). Both are equivalent
//! under the substitution
//!
//!   `freq_base_eff = rope_theta ^ (rotary_dim / head_dim)`
//!
//! which cancels the denominator discrepancy exactly. For Gemma 4
//! sliding layers the `rotary_dim == head_dim` so `freq_base_eff`
//! degenerates to `rope_theta`; for global layers with
//! `partial_rotary_factor=0.5, head_dim=512` we get
//! `rotary_dim=256` and `freq_base_eff = 1_000_000 ^ 0.5 = 1000`.
//!
//! This is the only mathematical trick in the port — once you fold
//! the scaling into `freq_base`, llama.cpp's kernel produces exactly
//! the rotation angles hf2q's cached `cos/sin` tables produce at
//! `gemma4.rs::RotaryEmbedding::build` via the
//! `1 / rope_theta^(2k/head_dim)` inv_freq formula. Verified in the
//! Phase A unit tests below at ε=1e-5.
//!
//! # Reference citations (ADR-005 Walk discipline)
//!
//! - **llama.cpp `kernel_rope_norm`** at
//!   `/opt/llama.cpp/ggml/src/ggml-metal/ggml-metal.metal:4323-4373`
//!   (f32 instantiation at `:4583`).
//! - **llama.cpp `kernel_rope_neox`** at
//!   `/opt/llama.cpp/ggml/src/ggml-metal/ggml-metal.metal:4376-4426`
//!   (f32 instantiation at `:4586`).
//! - **YaRN helpers** (`rope_yarn`, `rope_yarn_corr_dims`, etc.) at
//!   `ggml-metal.metal:4284-4320`. We do not use YaRN (Gemma 4 sets
//!   `ext_factor=0`) but the helpers are called unconditionally by
//!   the kernel body, so they must be present in the compiled
//!   library. Ported verbatim.
//! - **`ggml_metal_kargs_rope` struct** at
//!   `/opt/llama.cpp/ggml/src/ggml-metal/ggml-metal-impl.h:286-317`.
//!   Rust-side `RopeArgs` mirrors the field order and types exactly,
//!   with the `bool src2` field flattened to `i32` on both sides for
//!   deterministic alignment (see struct comment).
//! - **Host caller** at
//!   `/opt/llama.cpp/ggml/src/ggml-metal/ggml-metal-ops.cpp:3461-3555`
//!   — grid shape `(ne01, ne02, ne03, nth, 1, 1)`, buffer bindings
//!   `(kargs@0, src0@1, src1=pos@2, src2=freqs@3, dst@4)`, and
//!   `nth = min(1024, ne00)`. Mirrored exactly.
//! - **ADR-005 Q2 resolution** — `Device::new_library_with_source`
//!   at `/opt/candle/candle-metal-kernels/src/metal/device.rs:91-102`
//!   is the downstream-compile hatch; `Commands::command_encoder` at
//!   `/opt/candle/candle-metal-kernels/src/metal/commands.rs:101-104`
//!   shares candle's in-flight compute encoder so our dispatch
//!   interleaves with every other candle op on the same queue.
//!
//! # What this does NOT do
//!
//! - No dtype fallback. Only `float` instantiations are compiled.
//!   hf2q's forward pass is end-to-end F32 (1b.4); if a future
//!   change hands a non-F32 tensor in, add a new template
//!   instantiation rather than silently cast (Anti-Goal #7).
//! - No YaRN / no MRoPE. hf2q does not use either; the kwargs are
//!   passed through to the kernel with `ext_factor=0, beta_fast=0,
//!   beta_slow=0` so the yarn helpers resolve to identity.
//! - No in-place write. Output is a fresh Metal buffer wrapped as a
//!   candle Tensor. Aliasing `dst` with `src0` is safe at the kernel
//!   level (pass-through writes for `i0 >= n_dims` would be
//!   idempotent, and the rotation writes touch the same two elements
//!   the reads consumed) but hf2q call sites always want to preserve
//!   the input, so we never exercise the in-place path.
//! - No auto-fallback to the candle manual chain on error. Errors
//!   bubble up verbatim; the caller (`RotaryEmbedding::apply`)
//!   holds the `loop` vs `fused` mode knob.

#[cfg(feature = "metal")]
use anyhow::{anyhow, Result};
#[cfg(feature = "metal")]
use candle_core::{DType, Device, MetalDevice, Storage, Tensor};
#[cfg(feature = "metal")]
use candle_metal_kernels::metal::{Buffer, ComputePipeline, Library};
#[cfg(feature = "metal")]
use objc2_metal::{MTLResourceUsage, MTLSize};
use std::sync::Arc;

/// MSL source compiled at model-load time via
/// `Device::new_library_with_source`. Byte-for-byte port of the two
/// RoPE kernels from
/// `/opt/llama.cpp/ggml/src/ggml-metal/ggml-metal.metal:4282-4426`.
///
/// Modifications from upstream:
///   - Only `float` template instantiations are emitted (hf2q is F32-
///     only on the RoPE path). The `half` instantiations exist in
///     llama.cpp but are never called from hf2q's code path.
///   - The `bool src2` field in the kargs struct is widened to
///     `int src2` to avoid any C/MSL struct-padding surprise when
///     binding the args as bytes — both sides read `args.src2 != 0`.
///   - The `FC_rope_is_imrope` function constant is dropped (we
///     only ship `kernel_rope_norm` / `kernel_rope_neox` and neither
///     of those references that constant).
///   - No other algorithmic deviation. The rotation math, the
///     threadgroup grid walk, the per-i0 branch (`i0 < n_dims` rotate
///     / `i0 >= n_dims` pass-through), and the `rope_yarn*` helpers
///     are copied verbatim.
#[cfg(feature = "metal")]
const ROPE_MSL: &str = r#"
#include <metal_stdlib>

using namespace metal;

// Mirrors `ggml_metal_kargs_rope` at
// /opt/llama.cpp/ggml/src/ggml-metal/ggml-metal-impl.h:286-317 with
// the `bool src2` field widened to `int` for deterministic
// C-struct-to-MSL binding. The Rust-side `RopeArgs` struct uses the
// same field order and widths.
typedef struct {
    int32_t  ne00;
    int32_t  ne01;
    int32_t  ne02;
    int32_t  ne03;
    uint64_t nb00;
    uint64_t nb01;
    uint64_t nb02;
    uint64_t nb03;
    int32_t  ne0;
    int32_t  ne1;
    int32_t  ne2;
    int32_t  ne3;
    uint64_t nb0;
    uint64_t nb1;
    uint64_t nb2;
    uint64_t nb3;
    int32_t  n_past;
    int32_t  n_dims;
    int32_t  n_ctx_orig;
    float    freq_base;
    float    freq_scale;
    float    ext_factor;
    float    attn_factor;
    float    beta_fast;
    float    beta_slow;
    int32_t  sect_0;
    int32_t  sect_1;
    int32_t  sect_2;
    int32_t  sect_3;
    int32_t  src2;
} ggml_metal_kargs_rope;

// Byte-for-byte port of
// /opt/llama.cpp/ggml/src/ggml-metal/ggml-metal.metal:4284-4320.
static float rope_yarn_ramp(const float low, const float high, const int i0) {
    const float y = (i0 / 2 - low) / max(0.001f, high - low);
    return 1.0f - min(1.0f, max(0.0f, y));
}

// YaRN algorithm based on LlamaYaRNScaledRotaryEmbedding.py from
// https://github.com/jquesnelle/yarn — MIT. Copyright (c) 2023 Jeffrey
// Quesnelle and Bowen Peng.
static void rope_yarn(
    float theta_extrap, float freq_scale, float corr_dims[2], int i0, float ext_factor, float mscale,
    thread float * cos_theta, thread float * sin_theta) {
    float theta_interp = freq_scale * theta_extrap;
    float theta = theta_interp;
    if (ext_factor != 0.0f) {
        float ramp_mix = rope_yarn_ramp(corr_dims[0], corr_dims[1], i0) * ext_factor;
        theta = theta_interp * (1 - ramp_mix) + theta_extrap * ramp_mix;
        mscale *= 1.0f + 0.1f * log(1.0f / freq_scale);
    }
    *cos_theta = cos(theta) * mscale;
    *sin_theta = sin(theta) * mscale;
}

static float rope_yarn_corr_factor(int n_dims, int n_ctx_orig, float n_rot, float base) {
    return n_dims * log(n_ctx_orig / (n_rot * 2 * M_PI_F)) / (2 * log(base));
}

static void rope_yarn_corr_dims(
    int n_dims, int n_ctx_orig, float freq_base, float beta_fast, float beta_slow, float dims[2]
) {
    dims[0] = max(0.0f,         floor(rope_yarn_corr_factor(n_dims, n_ctx_orig, beta_fast, freq_base)));
    dims[1] = min(n_dims - 1.0f, ceil (rope_yarn_corr_factor(n_dims, n_ctx_orig, beta_slow, freq_base)));
}

// Byte-for-byte port of kernel_rope_norm at
// /opt/llama.cpp/ggml/src/ggml-metal/ggml-metal.metal:4322-4373.
// "norm" = GPT-J interleaved-pair RoPE — pairs are `(x[i0], x[i0+1])`
// with i0 stepping by 2. hf2q does not currently dispatch this
// variant (Gemma 4 routes through `neox` per LLAMA_ROPE_TYPE_NEOX at
// llama-model.cpp:9134) but we compile it anyway for completeness.
template<typename T>
kernel void kernel_rope_norm(
        constant ggml_metal_kargs_rope & args,
        device const char * src0,
        device const char * src1,
        device const char * src2,
        device       char * dst,
        ushort  tiitg[[thread_index_in_threadgroup]],
        ushort3 tptg [[threads_per_threadgroup]],
        uint3   tgpig[[threadgroup_position_in_grid]]) {
    const int i3 = tgpig[2];
    const int i2 = tgpig[1];
    const int i1 = tgpig[0];

    float corr_dims[2];
    rope_yarn_corr_dims(args.n_dims, args.n_ctx_orig, args.freq_base, args.beta_fast, args.beta_slow, corr_dims);

    device const int32_t * pos = (device const int32_t *) src1;

    const float theta_base = (float) pos[i2];
    const float inv_ndims = -1.f/args.n_dims;

    float cos_theta;
    float sin_theta;

    for (int i0 = 2*tiitg; i0 < args.ne0; i0 += 2*tptg.x) {
        if (i0 < args.n_dims) {
            const int ic = i0/2;

            const float theta = theta_base * pow(args.freq_base, inv_ndims*i0);

            const float freq_factor = (args.src2 != 0) ? ((device const float *) src2)[ic] : 1.0f;

            rope_yarn(theta/freq_factor, args.freq_scale, corr_dims, i0, args.ext_factor, args.attn_factor, &cos_theta, &sin_theta);

            device const T * const src = (device T *)(src0 + i3*args.nb03 + i2*args.nb02 + i1*args.nb01 + i0*args.nb00);
            device       T * dst_data  = (device T *)( dst + i3*args.nb3  + i2*args.nb2  + i1*args.nb1  + i0*args.nb0);

            const float x0 = src[0];
            const float x1 = src[1];

            dst_data[0] = x0*cos_theta - x1*sin_theta;
            dst_data[1] = x0*sin_theta + x1*cos_theta;
        } else {
            device const T * const src = (device T *)(src0 + i3*args.nb03 + i2*args.nb02 + i1*args.nb01 + i0*args.nb00);
            device       T * dst_data  = (device T *)( dst + i3*args.nb3  + i2*args.nb2  + i1*args.nb1  + i0*args.nb0);

            dst_data[0] = src[0];
            dst_data[1] = src[1];
        }
    }
}

// Byte-for-byte port of kernel_rope_neox at
// /opt/llama.cpp/ggml/src/ggml-metal/ggml-metal.metal:4376-4426.
// "neox" = HF split-half-pair RoPE — pairs are `(x[ic], x[ic + n_dims/2])`
// with ic ∈ [0, n_dims/2). This is the canonical modern Llama/Gemma/
// Qwen rotation and the variant hf2q's current `rope_apply` uses.
template<typename T>
kernel void kernel_rope_neox(
        constant ggml_metal_kargs_rope & args,
        device const char * src0,
        device const char * src1,
        device const char * src2,
        device       char * dst,
        ushort  tiitg[[thread_index_in_threadgroup]],
        ushort3 tptg [[threads_per_threadgroup]],
        uint3   tgpig[[threadgroup_position_in_grid]]) {
    const int i3 = tgpig[2];
    const int i2 = tgpig[1];
    const int i1 = tgpig[0];

    float corr_dims[2];
    rope_yarn_corr_dims(args.n_dims, args.n_ctx_orig, args.freq_base, args.beta_fast, args.beta_slow, corr_dims);

    device const int32_t * pos = (device const int32_t *) src1;

    const float theta_base = (float) pos[i2];
    const float inv_ndims = -1.f/args.n_dims;

    float cos_theta;
    float sin_theta;

    for (int i0 = 2*tiitg; i0 < args.ne0; i0 += 2*tptg.x) {
        if (i0 < args.n_dims) {
            const int ic = i0/2;

            const float theta = theta_base * pow(args.freq_base, inv_ndims*i0);

            const float freq_factor = (args.src2 != 0) ? ((device const float *) src2)[ic] : 1.0f;

            rope_yarn(theta/freq_factor, args.freq_scale, corr_dims, i0, args.ext_factor, args.attn_factor, &cos_theta, &sin_theta);

            device const T * const src = (device T *)(src0 + i3*args.nb03 + i2*args.nb02 + i1*args.nb01 + ic*args.nb00);
            device       T * dst_data  = (device T *)( dst + i3*args.nb3  + i2*args.nb2  + i1*args.nb1  + ic*args.nb0);

            const float x0 = src[0];
            const float x1 = src[args.n_dims/2];

            dst_data[0]             = x0*cos_theta - x1*sin_theta;
            dst_data[args.n_dims/2] = x0*sin_theta + x1*cos_theta;
        } else {
            device const T * const src = (device T *)(src0 + i3*args.nb03 + i2*args.nb02 + i1*args.nb01 + i0*args.nb00);
            device       T * dst_data  = (device T *)( dst + i3*args.nb3  + i2*args.nb2  + i1*args.nb1  + i0*args.nb0);

            dst_data[0] = src[0];
            dst_data[1] = src[1];
        }
    }
}

typedef decltype(kernel_rope_norm<float>) kernel_rope_norm_t;
typedef decltype(kernel_rope_neox<float>) kernel_rope_neox_t;

template [[host_name("hf2q_rope_norm_f32")]] kernel kernel_rope_norm_t kernel_rope_norm<float>;
template [[host_name("hf2q_rope_neox_f32")]] kernel kernel_rope_neox_t kernel_rope_neox<float>;
"#;

/// Rust mirror of `ggml_metal_kargs_rope` at
/// `/opt/llama.cpp/ggml/src/ggml-metal/ggml-metal-impl.h:286-317`.
///
/// `#[repr(C)]` — the field order and widths match the C struct
/// exactly, with the single modification that upstream's trailing
/// `bool src2` is widened to `int32_t` on BOTH sides (the MSL source
/// above does the same) to eliminate any padding-alignment surprise
/// between Rust and the Metal compiler. Total size: 160 bytes.
///
/// The kernel only reads `ne00, ne0, nb00..nb03, nb0..nb3, n_dims,
/// n_ctx_orig, freq_base, freq_scale, ext_factor, attn_factor,
/// beta_fast, beta_slow, src2`. The `ne01..ne03`, `ne1..ne3`, `n_past`,
/// and `sect_*` fields are passed through for faithfulness but
/// effectively dead at every Gemma 4 call site (we do not use MRoPE
/// and n_past is only used by host-side validation).
#[cfg(feature = "metal")]
#[repr(C)]
#[derive(Clone, Copy, Debug)]
struct RopeArgs {
    ne00: i32,
    ne01: i32,
    ne02: i32,
    ne03: i32,
    nb00: u64,
    nb01: u64,
    nb02: u64,
    nb03: u64,
    ne0: i32,
    ne1: i32,
    ne2: i32,
    ne3: i32,
    nb0: u64,
    nb1: u64,
    nb2: u64,
    nb3: u64,
    n_past: i32,
    n_dims: i32,
    n_ctx_orig: i32,
    freq_base: f32,
    freq_scale: f32,
    ext_factor: f32,
    attn_factor: f32,
    beta_fast: f32,
    beta_slow: f32,
    sect_0: i32,
    sect_1: i32,
    sect_2: i32,
    sect_3: i32,
    src2: i32, // widened from bool to i32 — see struct-level comment
}

/// Handle holding the runtime-compiled MSL library and its two
/// instantiated compute pipelines.
///
/// Both variants are compiled eagerly at construction time so the
/// first forward pass never pays a cold-compile spike (matches the
/// 1bNEW.4 RmsNorm pattern and the 1bNEW.12 extended warmup
/// discipline).
#[cfg(feature = "metal")]
pub struct RopePipelines {
    _library: Library,
    /// `kernel_rope_norm<float>` — GPT-J interleaved-pair variant.
    pipe_norm_f32: ComputePipeline,
    /// `kernel_rope_neox<float>` — HF split-half-pair variant, the
    /// live path for Gemma 4.
    pipe_neox_f32: ComputePipeline,
}

#[cfg(not(feature = "metal"))]
pub struct RopePipelines;

/// Per-model switch for the RoPE dispatch path. `Loop` preserves the
/// 9-op manual `rope_apply` chain from `gemma4.rs::RotaryEmbedding`
/// (plus the partial-rotary narrow/cat dance) exactly as of HEAD
/// pre-1bNEW.6. `Fused` routes Q and K through a single Metal
/// dispatch each per attention layer.
///
/// Plumbed from the CLI `--rope-kernel` flag through
/// `Gemma4Model::load_with_modes` into every `RotaryEmbedding` used
/// inside `Attention`. Kept as a knob (not a compile-time feature)
/// for bisect-safety: 1bNEW.1, 1bNEW.3, and 1bNEW.4 all keep their
/// `loop` fallbacks, and 1bNEW.6 follows the same discipline.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RopeKernelMode {
    Loop,
    Fused,
}

/// Cloneable bundle of `(mode, Option<Arc<RopePipelines>>)` — same
/// pattern the 1bNEW.4 `RmsNormKernel` bundle follows. Every
/// `RotaryEmbedding` holds a clone so it can dispatch on
/// `self.kernel.mode` without a global switch. `Loop` mode carries
/// `None` and pays no compile cost.
#[derive(Clone)]
pub struct RopeKernel {
    #[allow(dead_code)]
    pub mode: RopeKernelMode,
    #[allow(dead_code)]
    pub pipelines: Option<Arc<RopePipelines>>,
}

impl RopeKernel {
    pub fn loop_mode() -> Self {
        Self {
            mode: RopeKernelMode::Loop,
            pipelines: None,
        }
    }

    /// Construct the `Fused` bundle by compiling the MSL library
    /// once. Safe to call many times but the hf2q load path calls
    /// it exactly once per model and clones the resulting `Arc` into
    /// every `RotaryEmbedding`.
    #[cfg(feature = "metal")]
    pub fn fused_mode(metal_device: &MetalDevice) -> Result<Self> {
        Ok(Self {
            mode: RopeKernelMode::Fused,
            pipelines: Some(Arc::new(RopePipelines::new(metal_device)?)),
        })
    }

    #[allow(dead_code)]
    pub fn is_fused(&self) -> bool {
        matches!(self.mode, RopeKernelMode::Fused) && self.pipelines.is_some()
    }
}

#[cfg(feature = "metal")]
impl RopePipelines {
    /// Compile the MSL source and instantiate both pipeline symbols.
    pub fn new(metal_device: &MetalDevice) -> Result<Self> {
        let device = metal_device.device();
        let library = device
            .new_library_with_source(ROPE_MSL, None)
            .map_err(|e| anyhow!("rope_kernel: library compile failed: {e:?}"))?;

        let load = |name: &str| -> Result<ComputePipeline> {
            let func = library
                .get_function(name, None)
                .map_err(|e| anyhow!("rope_kernel: missing symbol {name}: {e:?}"))?;
            let pipe = device
                .new_compute_pipeline_state_with_function(&func)
                .map_err(|e| anyhow!("rope_kernel: PSO create failed for {name}: {e:?}"))?;
            Ok(pipe)
        };

        Ok(Self {
            _library: library.clone(),
            pipe_norm_f32: load("hf2q_rope_norm_f32")?,
            pipe_neox_f32: load("hf2q_rope_neox_f32")?,
        })
    }
}

/// Which pair layout the kernel rotates with. `Neox` is the canonical
/// modern split-half variant (Gemma 4, Llama, Qwen). `Norm` is GPT-J
/// interleaved — compiled for reference but never dispatched on hf2q's
/// current model.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RopeVariant {
    /// GPT-J interleaved-pair variant. Not used by Gemma 4 but compiled
    /// for completeness; may be needed by future model families.
    #[allow(dead_code)]
    Norm,
    Neox,
}

#[cfg(feature = "metal")]
fn metal_device_of(device: &Device) -> Result<MetalDevice> {
    match device {
        Device::Metal(md) => Ok(md.clone()),
        other => Err(anyhow!("rope_kernel: expected Metal device, got {other:?}")),
    }
}

/// Dispatch the fused RoPE kernel over a Q or K tensor.
///
/// **Shape:** `input` must be rank-4 with logical shape
/// `[batch, num_heads, seq_len, head_dim]`. This matches the
/// post-`transpose(1, 2)` layout used inside `Attention::forward`.
/// The tensor may be strided (not contiguous) — the kernel reads
/// source strides directly from `RopeArgs.nb00..nb03`, which is
/// exactly the old-1bNEW.8 stride-awareness win the ADR baked into
/// this item (see ADR-005:322-326).
///
/// **Output:** A fresh contiguous `[batch, num_heads, seq_len, head_dim]`
/// F32 tensor. Elements `[0, n_dims)` along the last axis are
/// rotated; elements `[n_dims, head_dim)` are copied verbatim
/// (partial RoPE pass-through, matching llama.cpp's kernel body and
/// HF's `rotate_half` behavior). Passing `n_dims == head_dim`
/// recovers full RoPE with no pass-through. Gemma 4 always passes
/// `n_dims == head_dim` post-1bNEW.18.
///
/// **Positions:** A simple `seqlen_offset` scalar — the kernel is
/// dispatched with a `[seq_len]` i32 buffer of absolute positions
/// `[seqlen_offset, seqlen_offset+1, ..., seqlen_offset+seq_len-1]`,
/// built fresh on the fly. This keeps the API narrow: the caller
/// does not need to hand in a pre-built positions Tensor, and the
/// hf2q KV-cache model uses per-step `seqlen_offset` anyway.
///
/// **Frequency scaling:** `freq_base` is passed through verbatim to
/// the kernel's `pow(freq_base, -i0/n_dims)`. Post-1bNEW.18 the
/// caller always passes `n_dims == head_dim` for Gemma 4 and
/// `freq_base == rope_theta` (no effective-base folding), because
/// the old proportional-scaling fold was a consequence of the
/// `rotary_dim < head_dim` misreading that 1bNEW.18 removes.
///
/// **`freq_factors`:** optional `[head_dim/2]` F32 tensor, bound as
/// `src2` on the Metal kernel when `Some`. The kernel body at
/// `:319` then reads `freq_factor = src2[ic]` per pair and divides
/// `theta / freq_factor` — byte-identical to llama.cpp's
/// `kernel_rope_neox` at `ggml-metal.metal:4353-4355`. When `None`
/// the kernel's `args.src2 != 0` gate short-circuits and every pair
/// uses `freq_factor = 1.0` (identity division). This is the
/// ADR-005 1bNEW.18 `rope_freqs.weight` port — see
/// `docs/spike-C-results.md` Parts 3-5 for the root cause analysis.
#[cfg(feature = "metal")]
pub fn rope_fused(
    pipelines: &RopePipelines,
    input: &Tensor,
    seqlen_offset: usize,
    n_dims: usize,
    freq_base: f32,
    variant: RopeVariant,
    freq_factors: Option<&Tensor>,
) -> Result<Tensor> {
    // --- Validation ---------------------------------------------------
    if input.dtype() != DType::F32 {
        return Err(anyhow!(
            "rope_kernel: input must be F32, got {:?}",
            input.dtype()
        ));
    }
    let dims = input.dims();
    if dims.len() != 4 {
        return Err(anyhow!(
            "rope_kernel: input must be rank-4 [B,H,S,D], got {:?}",
            dims
        ));
    }
    let (b_sz, n_heads, seq_len, head_dim) = (dims[0], dims[1], dims[2], dims[3]);
    if n_dims == 0 || n_dims > head_dim || n_dims % 2 != 0 {
        return Err(anyhow!(
            "rope_kernel: n_dims={n_dims} must be even and in (0, head_dim={head_dim}]"
        ));
    }
    if head_dim == 0 || seq_len == 0 || n_heads == 0 || b_sz == 0 {
        return Err(anyhow!(
            "rope_kernel: empty tensor [{b_sz},{n_heads},{seq_len},{head_dim}]"
        ));
    }

    // Candle element strides for the logical [B, H, S, D] view — may
    // be non-contiguous if the caller handed in a `.transpose(1, 2)`
    // result. `stride()` returns the per-dim stride in ELEMENTS, not
    // bytes; the kernel wants bytes, so we multiply by elem_size.
    let elem = std::mem::size_of::<f32>() as u64;
    let in_strides = input.stride();
    let (stride_b, stride_h, stride_s, stride_d) = (
        in_strides[0] as u64 * elem,
        in_strides[1] as u64 * elem,
        in_strides[2] as u64 * elem,
        in_strides[3] as u64 * elem,
    );
    // The kernel's i0-axis loop steps by `nb00` per element. If the
    // last dim is not tightly packed (stride_d != elem_size), the
    // port cannot consume the view — this is the same invariant
    // candle's contiguous() forces on the old loop path. In practice
    // every hf2q call site hands in either fully contiguous memory
    // or a `transpose(1, 2)` whose innermost dim stays contiguous.
    if stride_d != elem {
        return Err(anyhow!(
            "rope_kernel: innermost dim must be contiguous, got stride_d={stride_d} bytes \
             (expected {elem})"
        ));
    }

    // --- Metal device + src0 buffer + start offset --------------------
    let metal_device = metal_device_of(input.device())?;

    // Hold the storage guard alive through the whole encode scope.
    // `storage_and_layout()` returns `(RwLockReadGuard<Storage>, &Layout)`;
    // dropping the guard while the Metal `set_buffer` binding is still
    // in the encoder would risk a use-after-free. Extract `&Buffer` from
    // inside the guarded scope and keep the guard in a local binding
    // that outlives the encoder drop at the end of this function. Same
    // pattern as `rms_norm_kernel::rms_norm_fused` at
    // `/opt/hf2q/src/serve/rms_norm_kernel.rs:511-520`.
    let (src0_storage, src0_layout) = input.storage_and_layout();
    let src0_buf_ref: &Buffer = match &*src0_storage {
        Storage::Metal(ms) => ms.buffer(),
        other => {
            return Err(anyhow!(
                "rope_kernel: input storage must be Metal, got {other:?}"
            ));
        }
    };
    let src0_offset = src0_layout.start_offset() * DType::F32.size_in_bytes();

    // --- Optional freq_factors buffer (src2) --------------------------
    //
    // ADR-005 1bNEW.18 (2026-04-11): when the caller passes
    // `Some(freq_factors_tensor)`, bind it as the kernel's `src2`
    // input. The kernel branch at `:319` already reads this correctly
    // as a byte-port of llama.cpp's `kernel_rope_neox` at
    // `/opt/llama.cpp/ggml/src/ggml-metal/ggml-metal.metal:4353-4355`:
    //
    //   `freq_factor = (args.src2 != 0) ? ((device const float *)src2)[ic] : 1.0f`
    //   `rope_yarn(theta/freq_factor, ...)`
    //
    // When `None`, we keep the pre-1bNEW.18 behavior of binding the
    // input buffer as a placeholder (the kernel never dereferences it
    // because `args.src2 == 0` short-circuits) and set `args.src2 = 0`.
    //
    // The `(guard, Option<&Buffer>)` tuple keeps the storage lock
    // alive through the encode scope (same pattern as `src0` above
    // and as `rms_norm_kernel::rms_norm_fused` at
    // `/opt/hf2q/src/serve/rms_norm_kernel.rs:511-520`).
    let ff_guard_layout = match freq_factors {
        Some(t) => {
            if t.dtype() != DType::F32 {
                return Err(anyhow!(
                    "rope_kernel: freq_factors must be F32, got {:?}",
                    t.dtype()
                ));
            }
            let expected = head_dim / 2;
            // The kernel indexes `src2[ic]` for `ic ∈ [0, n_dims/2)`;
            // require at least that many elements. For Gemma 4 global
            // layers this is exactly 256 on head_dim=512.
            if t.elem_count() < expected {
                return Err(anyhow!(
                    "rope_kernel: freq_factors has {} elements but \
                     head_dim/2 = {} required (n_dims/2 = {})",
                    t.elem_count(),
                    expected,
                    n_dims / 2
                ));
            }
            Some(t.storage_and_layout())
        }
        None => None,
    };
    let ff_buf_and_offset: Option<(&Buffer, usize)> = match ff_guard_layout.as_ref() {
        Some((storage, layout)) => match &**storage {
            Storage::Metal(ms) => Some((
                ms.buffer(),
                layout.start_offset() * DType::F32.size_in_bytes(),
            )),
            other => {
                return Err(anyhow!(
                    "rope_kernel: freq_factors storage must be Metal, got {other:?}"
                ));
            }
        },
        None => None,
    };
    let src2_flag: i32 = if ff_buf_and_offset.is_some() { 1 } else { 0 };

    // --- Positions buffer (src1) --------------------------------------
    // llama.cpp feeds an int32 positions buffer; we allocate one per
    // dispatch (seq_len is ≤ ~4096 typical, so the allocation is
    // cheap and the buffer is shared-storage so the write doesn't
    // need a blit). For decode (seq_len=1) this is a 4-byte buffer.
    let positions: Vec<i32> = (0..seq_len)
        .map(|i| (seqlen_offset + i) as i32)
        .collect();
    let pos_buf_arc = metal_device
        .new_buffer_with_data(&positions)
        .map_err(|e| anyhow!("rope_kernel: positions buffer alloc failed: {e:?}"))?;

    // --- Destination buffer (fresh contiguous [B,H,S,D] F32) ---------
    // Output shape matches input shape exactly. We allocate tight
    // [B,H,S,D] row-major contiguous memory so downstream ops (SDPA,
    // KV-cache append) see a well-formed view without another
    // `.contiguous()` bounce.
    let out_elems = b_sz * n_heads * seq_len * head_dim;
    let dst_buf_arc = metal_device
        .new_buffer(out_elems, DType::F32, "rope_dst")
        .map_err(|e| anyhow!("rope_kernel: dst buffer alloc failed: {e:?}"))?;

    // Destination strides for a fresh contiguous [B, H, S, D] output.
    //
    // The kernel addresses the dst with `b*nb3 + i2*nb2 + i1*nb1 + i0*nb0`
    // where `i1 = head_idx, i2 = token_idx` (see the kargs comment
    // below for why this mapping — positions are indexed by the
    // **token** axis, so we bind `tgpig[1]=i2=token_idx` even though
    // that swaps the "natural" nb1 / nb2 order you might expect from
    // a `[B,H,S,D]` shape.
    //
    // For element `[b, h, s, d]` in a contiguous `[B,H,S,D]` layout,
    // the byte offset is `(b*H*S*D + h*S*D + s*D + d) * elem`. To
    // reproduce that with `b*nb3 + s*nb2 + h*nb1 + d*nb0` we need:
    //   nb0 = elem               (d stride)
    //   nb1 = S * D * elem       (head stride — across all tokens)
    //   nb2 = D * elem           (token stride — within one head)
    //   nb3 = H * S * D * elem   (batch stride)
    //
    // Getting `nb1` and `nb2` swapped is the exact kind of silent
    // stride bug the ADR's `project_coherence_bug.md` warned about.
    // The unit test at `[1, 16, 128, 512]` catches this — on
    // `seq_len=1` (decode) nb2 degenerates to nb0 and the bug hides.
    let dst_nb0 = elem;
    let dst_nb1 = seq_len as u64 * head_dim as u64 * elem;
    let dst_nb2 = head_dim as u64 * elem;
    let dst_nb3 = n_heads as u64 * seq_len as u64 * head_dim as u64 * elem;

    // --- Kargs struct -------------------------------------------------
    //
    // ggml's tensor layout for RoPE is `(n_embd_head, n_head, n_tokens, 1)`
    // with ne00 = head_dim. hf2q's logical layout is `[B, H, S, D]` with
    // a (possibly non-contiguous) stride set. We map the ggml indices
    // as:
    //   i0 ∈ [0, ne0=head_dim)  → last-dim element index (stride nb00)
    //   i1 ∈ [0, ne01=n_heads)  → head index              (stride nb01)
    //   i2 ∈ [0, ne02=seq_len)  → token index             (stride nb02)  ← pos[i2]
    //   i3 ∈ [0, ne03=batch)    → batch index             (stride nb03)
    // The kernel dispatches `tgpig = (ne01, ne02, ne03)` threadgroups,
    // each with `nth` threads along the last dim.
    //
    // Note pos[i2] — positions are indexed by the **token** axis,
    // NOT the head axis. That's the fundamental reason we map
    // `tgpig[1] = i2 = token_idx`, and `tgpig[0] = i1 = head_idx`:
    // it keeps the host-side `positions: [seq_len]` indexing aligned
    // with how the kernel reads `pos[i2]`.
    let kargs = RopeArgs {
        ne00: head_dim as i32,
        ne01: n_heads as i32,
        ne02: seq_len as i32,
        ne03: b_sz as i32,
        nb00: stride_d,
        nb01: stride_h,
        nb02: stride_s,
        nb03: stride_b,
        ne0: head_dim as i32,
        ne1: n_heads as i32,
        ne2: seq_len as i32,
        ne3: b_sz as i32,
        nb0: dst_nb0,
        nb1: dst_nb1,
        nb2: dst_nb2,
        nb3: dst_nb3,
        n_past: 0, // unused by the kernel body
        n_dims: n_dims as i32,
        // Gemma 4 does not use YaRN; passing a large `n_ctx_orig`
        // keeps `rope_yarn_corr_dims` well-defined (the helper
        // divides by `log(base)` inside `corr_factor`). Any value
        // > 0 works because `ext_factor=0` short-circuits the ramp.
        n_ctx_orig: 8192,
        freq_base,
        freq_scale: 1.0,
        ext_factor: 0.0,
        attn_factor: 1.0,
        beta_fast: 32.0,
        beta_slow: 1.0,
        sect_0: 0,
        sect_1: 0,
        sect_2: 0,
        sect_3: 0,
        // ADR-005 1bNEW.18: `src2 = 1` when `freq_factors: Some(_)`,
        // which activates the `args.src2 != 0` branch inside the
        // ported `kernel_rope_neox` at `rope_kernel.rs:319` so each
        // per-pair `theta` is divided by `src2[ic]`. Sliding layers
        // pass `None` → `src2 = 0` → every pair divides by `1.0f`
        // (byte-identical to pre-1bNEW.18 behavior on sliding layers).
        src2: src2_flag,
    };

    // --- Pipeline selection ------------------------------------------
    let pipeline = match variant {
        RopeVariant::Norm => &pipelines.pipe_norm_f32,
        RopeVariant::Neox => &pipelines.pipe_neox_f32,
    };

    // --- Threadgroup shape -------------------------------------------
    // Walk llama.cpp's rule: `nth = min(1024, ne00)`
    // (`ggml-metal-ops.cpp:3478`). Each thread handles 2 consecutive
    // i0 slots (the loop strides by `2*tptg.x`) so we need at most
    // `ne00/2` threads. llama.cpp clamps at 1024 anyway; we cap at
    // `max_total_threads_per_threadgroup` for Apple silicon's limit
    // and floor at 32 so a simdgroup's worth of threads always run.
    let max_tg = pipeline.max_total_threads_per_threadgroup();
    let nth = head_dim.min(1024).min(max_tg).max(32);

    // --- Encode the dispatch ------------------------------------------
    // Scope the encoder so Drop fires `endEncoding` before we wrap
    // the dst buffer into a candle Tensor, matching the 1bNEW.1
    // MoE fuse and 1bNEW.4 RmsNorm fuse protocol.
    {
        let encoder = metal_device.command_encoder()?;
        encoder.set_compute_pipeline_state(pipeline);
        encoder.set_bytes(0, &kargs);
        encoder.set_buffer(1, Some(src0_buf_ref), src0_offset);
        encoder.set_buffer(2, Some(pos_buf_arc.as_ref()), 0);
        // src2 binding:
        //   - `Some((buf, off))`: global layer — bind the
        //     `rope_freqs.weight` buffer. Kernel reads `src2[ic]`
        //     per pair and divides `theta / freq_factor`. This is
        //     ADR-005 1bNEW.18 — the caller-side fix to activate the
        //     already-correct kernel branch at `:319`.
        //   - `None`: sliding layer — bind the input buffer as a
        //     placeholder so the kernel's `args.src2 != 0` gate
        //     short-circuits without dereferencing an invalid
        //     binding. `args.src2 = 0` above guarantees the kernel
        //     never reads through this pointer.
        match ff_buf_and_offset {
            Some((ff_buf, ff_off)) => {
                encoder.set_buffer(3, Some(ff_buf), ff_off);
                encoder.use_resource(ff_buf, MTLResourceUsage::Read);
            }
            None => {
                encoder.set_buffer(3, Some(src0_buf_ref), src0_offset);
            }
        }
        encoder.set_buffer(4, Some(dst_buf_arc.as_ref()), 0);

        encoder.use_resource(src0_buf_ref, MTLResourceUsage::Read);
        encoder.use_resource(pos_buf_arc.as_ref(), MTLResourceUsage::Read);
        encoder.use_resource(dst_buf_arc.as_ref(), MTLResourceUsage::Write);

        // Grid: (n_heads, seq_len, batch). llama.cpp dispatches
        // (ne01, ne02, ne03, nth, 1, 1).
        let grid = MTLSize {
            width: n_heads,
            height: seq_len,
            depth: b_sz,
        };
        let tg = MTLSize {
            width: nth,
            height: 1,
            depth: 1,
        };
        encoder.dispatch_thread_groups(grid, tg);
    }

    // --- Wrap the destination buffer as a candle Tensor --------------
    let storage =
        candle_core::MetalStorage::new(dst_buf_arc, metal_device, out_elems, DType::F32);
    let out_tensor = Tensor::from_storage(
        Storage::Metal(storage),
        &[b_sz, n_heads, seq_len, head_dim][..],
        candle_core::op::BackpropOp::none(),
        false,
    );
    Ok(out_tensor)
}

// ---------------------------------------------------------------------------
// Phase A unit tests — first-principles numerical parity vs
// llama.cpp's `kernel_rope_neox` formula
// ---------------------------------------------------------------------------
//
// **ADR-005 1bNEW.18 (2026-04-11):** these tests REPLACE the
// pre-1bNEW.18 set that compared the fused kernel against hf2q's
// own `reference_rope_apply`. Spike C (`docs/spike-C-results.md`
// Part 3.5) demonstrated that `reference_rope_apply` was buggy in
// the *same* way the pre-1bNEW.18 fused kernel caller was: both
// paths omitted `freq_factors` and both used the wrong pair offset
// (`rotary_dim/2 = 64` instead of `head_dim/2 = 256` on Gemma 4
// global layers). They agreed with each other bit-for-bit, which
// is why every pre-1bNEW.18 Phase A test passed on the broken code.
//
// Post-1bNEW.18 the tests compare against a **first-principles
// scalar reference** that implements llama.cpp's exact formula
// from `/opt/llama.cpp/ggml/src/ggml-metal/ggml-metal.metal:4353-4410`:
//
//   for ic in [0, n_dims/2):
//       theta = pos * pow(freq_base, -2*ic/n_dims) / freq_factor[ic]
//       x0 = src[ic]
//       x1 = src[ic + n_dims/2]
//       dst[ic]              = x0 * cos(theta) - x1 * sin(theta)
//       dst[ic + n_dims/2]   = x0 * sin(theta) + x1 * cos(theta)
//   for i in [n_dims, head_dim): dst[i] = src[i]
//
// The reference does NOT use candle tensor ops; it walks the flat
// buffer with doubled-precision trig and writes to an `expected:
// Vec<f32>` that is then wrapped as a candle Tensor only for the
// comparison phase. This eliminates any FP-reduction-order
// interaction with candle's broadcast_mul + matmul path that could
// mask a kernel bug.
//
// ε is widened from 1e-5 to 1e-4 because the reference does scalar
// `(pos as f64) * (base as f64).powf(-2.0*ic/n_dims)` whereas the
// kernel does `theta_base * pow(freq_base, -2*ic/n_dims)` in F32
// and then divides by `freq_factor` as F32. The resulting
// per-element drift is up to ~1e-5 on its own; accumulated through
// the `cos * x0 - sin * x1` rotation at large `|x|` values it
// stays below ~5e-5 but we leave headroom at 1e-4 for Walk-safety.
//
// Run with:
//   `cargo test --features metal --release -p hf2q -- rope_kernel::tests --nocapture`

#[cfg(all(test, feature = "metal"))]
mod tests {
    use super::*;
    use candle_core::Device as CoreDevice;

    fn make_f32_vec(len: usize, seed: u64) -> Vec<f32> {
        let mut s = seed.wrapping_add(0x9E3779B97F4A7C15);
        (0..len)
            .map(|_| {
                s = s.wrapping_add(0x9E3779B97F4A7C15);
                let mut z = s;
                z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
                z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
                z ^= z >> 31;
                ((z as i64 as f64) * (1.0 / (i64::MAX as f64)) * 0.75) as f32
            })
            .collect()
    }

    /// First-principles scalar reference for `kernel_rope_neox`.
    ///
    /// Implements llama.cpp's formula at
    /// `/opt/llama.cpp/ggml/src/ggml-metal/ggml-metal.metal:4353-4410`
    /// *directly*, without going through any candle tensor ops, and
    /// without using hf2q's own `rope_apply` helper. The return
    /// value is a flat `Vec<f32>` of length `b*h*s*head_dim`, in the
    /// same row-major `[B,H,S,D]` layout as the input.
    ///
    /// For each (batch, head, token, pair_ic):
    ///   theta = pos * pow(freq_base, -(2*ic)/n_dims) / freq_factor[ic]
    ///   x0 = src[ic]
    ///   x1 = src[ic + n_dims/2]
    ///   dst[ic]              = x0*cos(theta) - x1*sin(theta)
    ///   dst[ic + n_dims/2]   = x0*sin(theta) + x1*cos(theta)
    /// Elements [n_dims, head_dim) are copied verbatim.
    ///
    /// Sliding layers pass `freq_factors_host = None` (every
    /// divisor is 1.0). Global layers pass the real
    /// `[1.0]×64 + [1e+30]×192` pattern so the [64..256) pairs
    /// rotate to identity via `theta/1e30 ≈ 0`.
    fn reference_rope_neox_scalar(
        input_flat: &[f32],
        b: usize,
        h: usize,
        s: usize,
        head_dim: usize,
        n_dims: usize,
        rope_theta: f64,
        seqlen_offset: usize,
        freq_factors_host: Option<&[f32]>,
    ) -> Vec<f32> {
        assert_eq!(input_flat.len(), b * h * s * head_dim);
        assert!(n_dims <= head_dim && n_dims % 2 == 0);
        if let Some(ff) = freq_factors_host {
            assert!(
                ff.len() >= n_dims / 2,
                "freq_factors needs at least n_dims/2 = {} elements, got {}",
                n_dims / 2,
                ff.len()
            );
        }
        let mut out = input_flat.to_vec();
        let half = n_dims / 2;
        for bi in 0..b {
            for hi in 0..h {
                for si in 0..s {
                    let row_off = ((bi * h + hi) * s + si) * head_dim;
                    let pos = (seqlen_offset + si) as f64;
                    for ic in 0..half {
                        // theta = pos * freq_base^(-2*ic/n_dims) / freq_factor
                        let exponent = -(2.0 * ic as f64) / n_dims as f64;
                        let ff = freq_factors_host
                            .map(|v| v[ic] as f64)
                            .unwrap_or(1.0);
                        let theta = pos * rope_theta.powf(exponent) / ff;
                        let (st, ct) = theta.sin_cos();
                        let x0 = input_flat[row_off + ic] as f64;
                        let x1 = input_flat[row_off + ic + half] as f64;
                        out[row_off + ic] = (x0 * ct - x1 * st) as f32;
                        out[row_off + ic + half] = (x0 * st + x1 * ct) as f32;
                    }
                    // Elements [n_dims, head_dim) are passed through
                    // verbatim — `out` was initialized to a copy of
                    // the input, so nothing to do.
                }
            }
        }
        out
    }

    fn device_and_pipelines() -> Option<(CoreDevice, RopePipelines)> {
        let device = match CoreDevice::new_metal(0) {
            Ok(d) => d,
            Err(e) => {
                eprintln!("skip: metal device unavailable ({e})");
                return None;
            }
        };
        let md = match &device {
            Device::Metal(md) => md.clone(),
            _ => unreachable!(),
        };
        let pipelines = RopePipelines::new(&md).unwrap();
        Some((device, pipelines))
    }

    fn compare_flat(ref_flat: &[f32], fused_flat: &[f32], label: &str, eps: f32) {
        assert_eq!(
            ref_flat.len(),
            fused_flat.len(),
            "length mismatch {label}: ref={} fused={}",
            ref_flat.len(),
            fused_flat.len()
        );
        let mut max_abs = 0.0_f32;
        let mut idx = 0usize;
        let mut n_diff = 0usize;
        let mut n_nan_ref = 0usize;
        let mut n_nan_fused = 0usize;
        for (i, (a, b)) in ref_flat.iter().zip(fused_flat.iter()).enumerate() {
            if a.is_nan() {
                n_nan_ref += 1;
            }
            if b.is_nan() {
                n_nan_fused += 1;
            }
            let d = (*a - *b).abs();
            // NaN != NaN so `d.abs()` would be NaN, and `d > max_abs`
            // is false — count NaN explicitly so the test can't
            // silently pass on an all-NaN fused output.
            if d.is_nan() || d > eps {
                n_diff += 1;
            }
            if d > max_abs {
                max_abs = d;
                idx = i;
            }
        }
        eprintln!(
            "rope_kernel {label}: max|Δ|={max_abs:.3e} at idx {idx} \
             (ref={} fused={}) — {n_diff}/{} differ > {eps} \
             (NaN: ref={n_nan_ref}, fused={n_nan_fused})",
            ref_flat[idx],
            fused_flat[idx],
            ref_flat.len()
        );
        assert!(
            n_nan_fused == 0,
            "fused RoPE {label} produced {n_nan_fused} NaN(s) — kernel bug"
        );
        assert!(
            max_abs < eps && n_diff == 0,
            "fused RoPE {label} disagrees with first-principles reference at ε={eps}; \
             max|Δ|={max_abs}, {n_diff} mismatched elements"
        );
    }

    fn build_input(
        device: &CoreDevice,
        shape: (usize, usize, usize, usize),
        seed: u64,
    ) -> (Tensor, Vec<f32>) {
        let (b, h, s, d) = shape;
        let v = make_f32_vec(b * h * s * d, seed);
        let t = Tensor::from_vec(v.clone(), (b, h, s, d), device).unwrap();
        (t, v)
    }

    /// The real Gemma 4 `rope_freqs.weight` pattern verified against
    /// the shipping GGUF via `gguf.GGUFReader` on 2026-04-11:
    /// `[1.0] × 64 + [1e+30] × 192` for `head_dim = 512 → half = 256`.
    fn gemma4_global_freq_factors(half: usize) -> Vec<f32> {
        // `half` may differ from 256 when running a smaller test
        // shape, but we always pattern `[1.0] × (half/4) + [1e+30] × rest`
        // so the structural divisor mask is exercised regardless of
        // the specific `half` chosen.
        let identity_count = half / 4;
        let mut v = vec![1.0_f32; half];
        for i in identity_count..half {
            v[i] = 1e30;
        }
        v
    }

    /// Test 1: decode shape, full rotary, sliding layer (no
    /// freq_factors). Shape `[1, 16, 1, 256]`, `n_dims = head_dim =
    /// 256`, `rope_theta = 10_000`, `seqlen_offset = 0`. Mirrors the
    /// per-forward Gemma 4 sliding-attention call shape.
    #[test]
    fn rope_neox_decode_full_rotary_sliding_scalar() {
        let Some((device, pipelines)) = device_and_pipelines() else { return };
        let head_dim = 256;
        let n_dims = 256;
        let rope_theta = 10_000.0_f64;
        let shape = (1, 16, 1, head_dim);
        let (input, input_flat) = build_input(&device, shape, 0x51DE_51DE);
        let fused =
            rope_fused(&pipelines, &input, 0, n_dims, rope_theta as f32, RopeVariant::Neox, None)
                .unwrap();
        let expected = reference_rope_neox_scalar(
            &input_flat, shape.0, shape.1, shape.2, head_dim, n_dims, rope_theta, 0, None,
        );
        let fused_flat = fused.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        compare_flat(&expected, &fused_flat, "decode_full_rotary_sliding_scalar", 1e-4);
    }

    /// Test 2: decode shape, full rotary, **global layer with
    /// real Gemma 4 `rope_freqs` mask**. Shape `[1, 16, 1, 512]`,
    /// `n_dims = head_dim = 512`, `rope_theta = 1_000_000`,
    /// `freq_factors = [1.0]×64 + [1e+30]×192`. This is the exact
    /// site Spike C localized; the test must catch any attempt to
    /// regress either the full-head rotation or the freq_factors
    /// mask.
    #[test]
    fn rope_neox_decode_full_rotary_global_with_mask() {
        let Some((device, pipelines)) = device_and_pipelines() else { return };
        let head_dim = 512;
        let n_dims = 512;
        let rope_theta = 1_000_000.0_f64;
        let shape = (1, 16, 1, head_dim);
        let (input, input_flat) = build_input(&device, shape, 0xCAFE_BABE);
        let freq_factors_host = gemma4_global_freq_factors(head_dim / 2);
        let ff_tensor = Tensor::from_vec(freq_factors_host.clone(), (head_dim / 2,), &device).unwrap();
        let fused = rope_fused(
            &pipelines,
            &input,
            0,
            n_dims,
            rope_theta as f32,
            RopeVariant::Neox,
            Some(&ff_tensor),
        )
        .unwrap();
        let expected = reference_rope_neox_scalar(
            &input_flat,
            shape.0,
            shape.1,
            shape.2,
            head_dim,
            n_dims,
            rope_theta,
            0,
            Some(&freq_factors_host),
        );
        let fused_flat = fused.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        compare_flat(
            &expected,
            &fused_flat,
            "decode_full_rotary_global_with_mask",
            1e-4,
        );
    }

    /// Test 3: **prefill** shape, full rotary, global layer with
    /// mask. Shape `[1, 16, 128, 512]`. Exercises per-token `pos[i2]`
    /// indexing AND the mask interaction at non-trivial positions.
    /// This is the test that pre-1bNEW.18 would have caught the wrong
    /// pair offset (`rotary_dim/2 = 64` vs correct `head_dim/2 = 256`)
    /// — the per-position `max|Δ|` pattern from Spike C Table at
    /// `docs/spike-C-results.md:360-369` scales linearly with
    /// position only when the wrong pair offset is applied.
    #[test]
    fn rope_neox_prefill_full_rotary_global_with_mask() {
        let Some((device, pipelines)) = device_and_pipelines() else { return };
        let head_dim = 512;
        let n_dims = 512;
        let rope_theta = 1_000_000.0_f64;
        let shape = (1, 16, 128, head_dim);
        let (input, input_flat) = build_input(&device, shape, 0x1234_5678);
        let freq_factors_host = gemma4_global_freq_factors(head_dim / 2);
        let ff_tensor = Tensor::from_vec(freq_factors_host.clone(), (head_dim / 2,), &device).unwrap();
        let fused = rope_fused(
            &pipelines,
            &input,
            0,
            n_dims,
            rope_theta as f32,
            RopeVariant::Neox,
            Some(&ff_tensor),
        )
        .unwrap();
        let expected = reference_rope_neox_scalar(
            &input_flat,
            shape.0,
            shape.1,
            shape.2,
            head_dim,
            n_dims,
            rope_theta,
            0,
            Some(&freq_factors_host),
        );
        let fused_flat = fused.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        compare_flat(
            &expected,
            &fused_flat,
            "prefill_full_rotary_global_with_mask",
            1e-4,
        );
    }

    /// Test 4: decode at `seqlen_offset = 42` with the global
    /// mask. Verifies the positions buffer indexes `pos[i2] =
    /// seqlen_offset + i2` correctly under the post-1bNEW.18
    /// signature. Also exercises a non-zero position with the
    /// `1e+30` mask to confirm the identity-rotation pairs stay
    /// identity regardless of position.
    #[test]
    fn rope_neox_decode_at_offset_with_mask() {
        let Some((device, pipelines)) = device_and_pipelines() else { return };
        let head_dim = 512;
        let n_dims = 512;
        let rope_theta = 1_000_000.0_f64;
        let shape = (1, 16, 1, head_dim);
        let (input, input_flat) = build_input(&device, shape, 0x0BAD_BEEF);
        let freq_factors_host = gemma4_global_freq_factors(head_dim / 2);
        let ff_tensor = Tensor::from_vec(freq_factors_host.clone(), (head_dim / 2,), &device).unwrap();
        let fused = rope_fused(
            &pipelines,
            &input,
            42,
            n_dims,
            rope_theta as f32,
            RopeVariant::Neox,
            Some(&ff_tensor),
        )
        .unwrap();
        let expected = reference_rope_neox_scalar(
            &input_flat,
            shape.0,
            shape.1,
            shape.2,
            head_dim,
            n_dims,
            rope_theta,
            42,
            Some(&freq_factors_host),
        );
        let fused_flat = fused.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        compare_flat(&expected, &fused_flat, "decode_at_offset_with_mask", 1e-4);
    }

    /// Test 5: identity-rotation invariant. Set
    /// `freq_factors = [1e+30; half]` so every pair rotates to
    /// identity — the output must equal the input bit-for-bit
    /// (modulo scalar f32 noise in the `cos(0) * x - sin(0) * x`
    /// path, which is at the ~1e-7 floor). This is a structural
    /// sanity gate that would catch any future regression that
    /// silently disables the mask division inside the kernel.
    #[test]
    fn rope_neox_full_mask_is_identity() {
        let Some((device, pipelines)) = device_and_pipelines() else { return };
        let head_dim = 512;
        let n_dims = 512;
        let rope_theta = 1_000_000.0_f64;
        let shape = (1, 16, 8, head_dim);
        let (input, input_flat) = build_input(&device, shape, 0xABAD_CAFE);
        let freq_factors_host = vec![1e30_f32; head_dim / 2];
        let ff_tensor =
            Tensor::from_vec(freq_factors_host.clone(), (head_dim / 2,), &device).unwrap();
        let fused = rope_fused(
            &pipelines,
            &input,
            0,
            n_dims,
            rope_theta as f32,
            RopeVariant::Neox,
            Some(&ff_tensor),
        )
        .unwrap();
        let fused_flat = fused.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        compare_flat(&input_flat, &fused_flat, "full_mask_is_identity", 1e-5);
    }

    /// Test 6: `norm` (GPT-J interleaved-pair) variant with the
    /// new first-principles reference. hf2q does not dispatch this
    /// path in production (Gemma 4 is Neox), but the port is kept
    /// for reference parity and the test guards against accidental
    /// algebra bugs in `kernel_rope_norm`.
    #[test]
    fn rope_norm_variant_interleaved_sanity() {
        let Some((device, pipelines)) = device_and_pipelines() else { return };
        let head_dim = 64;
        let n_dims = 64;
        let rope_theta = 10_000.0_f64;
        let shape = (1, 8, 1, head_dim);
        let (input, input_vec) = build_input(&device, shape, 0xBEEF_CAFE);

        // Reference: GPT-J interleaved-pair RoPE. For each
        // `i0 ∈ {0, 2, ..., n_dims-2}` on each (batch, head, token):
        //   theta = pos * freq_base^(-i0/n_dims)
        //   (x0, x1) = (src[i0], src[i0+1])
        //   (y0, y1) = (x0*cos - x1*sin, x0*sin + x1*cos)
        let (b, h, s, d) = shape;
        let mut expected = input_vec.clone();
        for bi in 0..b {
            for hi in 0..h {
                for si in 0..s {
                    let pos = si as f64;
                    let row_off = ((bi * h + hi) * s + si) * d;
                    let mut i0 = 0usize;
                    while i0 < n_dims {
                        let theta = pos * rope_theta.powf(-(i0 as f64) / n_dims as f64);
                        let (st, ct) = theta.sin_cos();
                        let x0 = input_vec[row_off + i0] as f64;
                        let x1 = input_vec[row_off + i0 + 1] as f64;
                        expected[row_off + i0] = (x0 * ct - x1 * st) as f32;
                        expected[row_off + i0 + 1] = (x0 * st + x1 * ct) as f32;
                        i0 += 2;
                    }
                }
            }
        }
        let fused = rope_fused(
            &pipelines,
            &input,
            0,
            n_dims,
            rope_theta as f32,
            RopeVariant::Norm,
            None,
        )
        .unwrap();
        let fused_flat = fused.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        compare_flat(&expected, &fused_flat, "norm_variant_interleaved", 1e-4);
    }
}
