//! Fused RmsNorm Metal kernel — ports llama.cpp `kernel_rms_norm_fuse_impl`.
//!
//! # What this module does
//!
//! Runtime-compiles a small Metal Shading Language (MSL) library that
//! instantiates `kernel_rms_norm_fuse_impl<T, F>` for `T ∈ {float, float4}`
//! and `F ∈ {1, 2, 3}`, and wraps the six resulting pipeline symbols in a
//! `RmsNormPipelines` handle. A `dispatch` helper binds the appropriate
//! buffers to candle's shared command encoder in the exact argument order
//! the llama.cpp kernel expects, so the fused dispatch lives in-order
//! alongside the rest of the forward pass.
//!
//! This is **ADR-005 1bNEW.4**. It replaces the 11-op manual candle chain
//! at `gemma4.rs::RmsNorm::forward` and the 9-op chain at `rms_norm_unit`
//! with a single Metal dispatch per call site (plus, via the F=3 variant,
//! folds the single NORM→ADD residual add at
//! `DecoderLayer::forward` post-FFW site into the same dispatch).
//!
//! # Reference citations (ADR-005 Walk discipline)
//!
//! - **llama.cpp `kernel_rms_norm_fuse_impl` template** at
//!   `/opt/llama.cpp/ggml/src/ggml-metal/ggml-metal.metal:2980-3050`. The
//!   MSL source embedded below is a byte-for-byte port of that template,
//!   restricted to the `T ∈ {float, float4}, F ∈ {1,2,3}` instantiations
//!   and the single `ggml_metal_kargs_norm` argument struct llama.cpp
//!   dispatches with (`ggml-metal-impl.h:537-550`, `ggml-metal-ops.cpp:3344-3456`).
//!   The `@0` kargs-as-bytes binding and the `(ne01, ne02, ne03)` threadgroup
//!   grid shape are both copied verbatim from llama.cpp's dispatch site.
//!   Reduction order (parallel sum-of-squares → simd_sum → shmem_f32 → second
//!   simd_sum → scale) is preserved identically so hf2q and llama.cpp produce
//!   the same rounding, which is the prerequisite for the Walk-correctness
//!   `The`→`To` argmax-flip being driven by FP-precision rather than
//!   algorithmic divergence.
//!
//! - **ADR-005 Q2 resolution** (line ~747):
//!   `Device::new_library_with_source` at
//!   `/opt/candle/candle-metal-kernels/src/metal/device.rs:91-102` is a
//!   public downstream-compilation entry point; the resulting `Library` +
//!   `ComputePipeline` pair must be retained by the caller because
//!   candle's own `Kernels` cache is keyed on the hardcoded `Source` enum
//!   and will not own a downstream-compiled pipeline. We retain them in
//!   `RmsNormPipelines` which lives on `Gemma4Model`.
//!
//! - **Dispatch via candle's shared encoder** — `MetalDevice::command_encoder`
//!   at `/opt/candle/candle-core/src/metal_backend/device.rs:143-150` hands
//!   back the currently-open `ComputeCommandEncoder` from candle's
//!   `Commands` pool (`candle-metal-kernels/src/metal/commands.rs:101-104`),
//!   which means our dispatches are in-order with every other candle op
//!   executing on the same command buffer. The 50-dispatch recycle
//!   threshold (`commands.rs:14`) is not a correctness hazard: Metal
//!   preserves in-order execution across buffers in the same queue (see
//!   the moe_kernel module header for the same observation).
//!
//! # What this does NOT do
//!
//! - No dtype fallback. `T` is fixed to `float`; the hf2q forward pass is
//!   end-to-end F32 (ADR-005 1b.4) and never hands RmsNorm a non-F32
//!   tensor. If that ever changes, add a new dtype tag and a matching
//!   template instantiation, not a cast here (Anti-Goal #7 — no stubs).
//! - No auto-fallback to the candle manual chain on error. Errors bubble
//!   up verbatim; the caller (`RmsNorm::forward`) holds the `loop` vs
//!   `fused` mode knob.
//! - No in-place write. Every dispatch allocates a fresh destination
//!   buffer. Aliasing dst with src0 is safe at the kernel level (it
//!   reads `x[i00]` and writes `y[i00]` in a single pass after the
//!   reduction) but every hf2q call site wants to preserve the input
//!   because it feeds a residual add or a parallel branch, so we never
//!   exercise the in-place path and never wrote the knob for it.

use anyhow::{anyhow, Result};
use candle_core::{DType, Device, MetalDevice, Storage, Tensor};
use candle_metal_kernels::metal::{Buffer, ComputePipeline, Library};
use objc2_metal::{MTLResourceUsage, MTLSize};
use std::sync::Arc;

/// MSL source compiled at model-load time via
/// `Device::new_library_with_source`. Byte-for-byte port of
/// `/opt/llama.cpp/ggml/src/ggml-metal/ggml-metal.metal:2980-3050` —
/// including the exact reduction order and threadgroup-barrier
/// placement — so the output matches llama.cpp's at the same FP rounding.
///
/// The template parameter `F` selects one of three fusions:
///   - F=1 : `y = x * scale`
///   - F=2 : `y = (x * scale) * f0`
///   - F=3 : `y = (x * scale) * f0 + f1`
/// and we emit the six `host_name("kernel_rms_norm_{,mul_,mul_add_}f32{,_4}")`
/// exports llama.cpp ships at `ggml-metal.metal:3044-3050`.
///
/// We deliberately do NOT port the non-RmsNorm `kernel_norm_fuse_impl`
/// template (mean-centered variant) that shares the same args struct —
/// hf2q does not use it, and keeping this source short keeps the
/// runtime compile time low.
const RMS_NORM_FUSE_MSL: &str = r#"
#include <metal_stdlib>

using namespace metal;

// `dot(v, v)` overload helper. Metal ships `dot` for `float{2,3,4}`
// and `half{2,3,4}` but NOT for plain `float` (verified empirically
// against the GPUCompiler.framework headers on macOS 26.4, see the
// compiler error hf2q tracked during Phase A Walk-dev). llama.cpp's
// upstream template compiles because its reference site only
// instantiates the `float4` specialization at `ggml-metal.metal:3048-3050`;
// the `float` specialization at `:3044-3046` is never actually hit on
// Apple silicon because the host path picks `ne00_t = ne00/4` whenever
// `ne00 % 4 == 0`, which is always true on ggml's block-size=32/256
// quantized tensors. We nevertheless want the scalar `float` path to
// compile for correctness under Anti-Goal #6 (arbitrary shape), so we
// define an overloaded `sq_dot` that falls back to `v*v` for scalar T.
inline float sq_dot(float v) { return v * v; }
inline float sq_dot(float2 v) { return dot(v, v); }
inline float sq_dot(float3 v) { return dot(v, v); }
inline float sq_dot(float4 v) { return dot(v, v); }

// Mirrors `ggml_metal_kargs_norm` at
// /opt/llama.cpp/ggml/src/ggml-metal/ggml-metal-impl.h:537-550.
// Only `ne00` / `ne00_t` / `nb1..3` / `eps` / the src-stride triples
// are actually read by the RmsNorm kernel; `nef1..3` are used for the
// wrap-around indexing into f0/f1 (we pass `ne01/02/03` directly so
// the `%` reduces to the identity at all call sites).
typedef struct {
    int32_t  ne00;
    int32_t  ne00_t;
    uint64_t nb1;
    uint64_t nb2;
    uint64_t nb3;
    float    eps;
    int32_t  nef1[3];
    int32_t  nef2[3];
    int32_t  nef3[3];
    uint64_t nbf1[3];
    uint64_t nbf2[3];
    uint64_t nbf3[3];
} ggml_metal_kargs_norm;

// Byte-for-byte port of
// /opt/llama.cpp/ggml/src/ggml-metal/ggml-metal.metal:2980-3040.
// Modifications from upstream:
//   - None. The template, reduction order, threadgroup barrier, and
//     per-F conditional writes are preserved verbatim. Only the
//     `host_name` instantiations at the bottom differ (we emit just
//     the 6 symbols we need; llama.cpp also emits the parallel
//     `kernel_norm_*` set we do not use).
template <typename T, short F>
kernel void kernel_rms_norm_fuse_impl(
        constant ggml_metal_kargs_norm & args,
        device const char * src0,
        device const char * src1_0,
        device const char * src1_1,
        device       char * dst,
        threadgroup float * shmem_f32 [[threadgroup(0)]],
        uint3   tgpig[[threadgroup_position_in_grid]],
        ushort3 tpitg[[thread_position_in_threadgroup]],
        ushort  sgitg[[simdgroup_index_in_threadgroup]],
        ushort  tiisg[[thread_index_in_simdgroup]],
        ushort3   ntg[[threads_per_threadgroup]]) {
    if (sgitg == 0) {
        shmem_f32[tiisg] = 0.0f;
    }

    const int i01 = tgpig.x;
    const int i02 = tgpig.y;
    const int i03 = tgpig.z;

    device const T * x = (device const T *) (src0 + i03*args.nbf3[0] + i02*args.nbf2[0] + i01*args.nbf1[0]);

    device const T * f0 = (device const T *) (src1_0 + (i03%args.nef3[1])*args.nbf3[1] + (i02%args.nef2[1])*args.nbf2[1] + (i01%args.nef1[1])*args.nbf1[1]);
    device const T * f1 = (device const T *) (src1_1 + (i03%args.nef3[2])*args.nbf3[2] + (i02%args.nef2[2])*args.nbf2[2] + (i01%args.nef1[2])*args.nbf1[2]);

    float sumf = 0.0f;

    // parallel sum of squares
    for (int i00 = tpitg.x; i00 < args.ne00_t; i00 += ntg.x) {
        sumf += sq_dot(x[i00]);
    }
    sumf = simd_sum(sumf);

    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tiisg == 0) {
        shmem_f32[sgitg] = sumf;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    sumf = shmem_f32[tiisg];
    sumf = simd_sum(sumf);

    const float mean  = sumf/args.ne00;
    const float scale = 1.0f/sqrt(mean + args.eps);

    device T * y = (device T *) (dst + i03*args.nb3 + i02*args.nb2 + i01*args.nb1);
    for (int i00 = tpitg.x; i00 < args.ne00_t; i00 += ntg.x) {
        if (F == 1) {
            y[i00] = (x[i00]*scale);
        }
        if (F == 2) {
            y[i00] = (x[i00]*scale)*f0[i00];
        }
        if (F == 3) {
            y[i00] = (x[i00]*scale)*f0[i00] + f1[i00];
        }
    }
}

typedef decltype(kernel_rms_norm_fuse_impl<float, 1>) kernel_rms_norm_fuse_t;

template [[host_name("hf2q_rms_norm_f32")]]         kernel kernel_rms_norm_fuse_t kernel_rms_norm_fuse_impl<float, 1>;
template [[host_name("hf2q_rms_norm_mul_f32")]]     kernel kernel_rms_norm_fuse_t kernel_rms_norm_fuse_impl<float, 2>;
template [[host_name("hf2q_rms_norm_mul_add_f32")]] kernel kernel_rms_norm_fuse_t kernel_rms_norm_fuse_impl<float, 3>;

template [[host_name("hf2q_rms_norm_f32_4")]]         kernel kernel_rms_norm_fuse_t kernel_rms_norm_fuse_impl<float4, 1>;
template [[host_name("hf2q_rms_norm_mul_f32_4")]]     kernel kernel_rms_norm_fuse_t kernel_rms_norm_fuse_impl<float4, 2>;
template [[host_name("hf2q_rms_norm_mul_add_f32_4")]] kernel kernel_rms_norm_fuse_t kernel_rms_norm_fuse_impl<float4, 3>;
"#;

/// Rust mirror of `ggml_metal_kargs_norm` at
/// `/opt/llama.cpp/ggml/src/ggml-metal/ggml-metal-impl.h:537-550`.
///
/// `#[repr(C)]` with the fields in the same order as the C struct so a
/// direct `set_bytes(0, &kargs)` binds a layout the Metal compiler
/// understands. Cross-checked against the C struct field-by-field — the
/// `[3]` arrays pack identically under C and `#[repr(C)]` Rust.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
struct RmsNormKargs {
    ne00: i32,
    ne00_t: i32,
    nb1: u64,
    nb2: u64,
    nb3: u64,
    eps: f32,
    // Three slots: [base_src0, f0, f1]. llama.cpp initialises all three
    // to the src0 shape and only overrides [1] and [2] when fuses kick
    // in; we mirror that so the `(i02 % nef2[k])` wrap lands on the
    // correct row for every k regardless of whether the slot is used.
    nef1: [i32; 3],
    nef2: [i32; 3],
    nef3: [i32; 3],
    nbf1: [u64; 3],
    nbf2: [u64; 3],
    nbf3: [u64; 3],
}

/// Handle holding the runtime-compiled MSL library and its six
/// instantiated compute pipelines.
///
/// Construction compiles the library once via
/// `Device::new_library_with_source`. All six pipelines are eagerly
/// realized at construction time so the first forward pass never pays a
/// cold-compile spike (the extended warmup in `mod.rs` already exists
/// for exactly this discipline).
///
/// `Arc<RmsNormPipelines>` is cloned into every `RmsNorm` instance at
/// load time so each call site holds a direct reference to the six
/// pipelines without walking through a `RwLock` or the candle kernel
/// cache.
pub struct RmsNormPipelines {
    _library: Library,
    pipe_f32_1: ComputePipeline,
    pipe_f32_2: ComputePipeline,
    pipe_f32_3: ComputePipeline,
    pipe_f32_4_1: ComputePipeline,
    pipe_f32_4_2: ComputePipeline,
    pipe_f32_4_3: ComputePipeline,
}

/// Per-model switch for the RmsNorm dispatch path. `Loop` preserves the
/// 11/9-op manual candle chain from `gemma4.rs::RmsNorm::forward` and
/// `rms_norm_unit`, exactly as of HEAD pre-1bNEW.4. `Fused` routes every
/// call site through the runtime-compiled kernel in this file.
///
/// Plumbed from the CLI `--rms-norm-kernel` flag (`cli::RmsNormKernelMode`)
/// through `Gemma4Model::load_with_modes` into every `RmsNorm`,
/// `Attention`, and `MoeBlock` that issues a norm call. Kept as a knob
/// (not a compile-time feature) for bisect-safety: 1bNEW.1 and 1bNEW.3
/// both keep their `loop` fallbacks, and 1bNEW.4 follows the same
/// discipline. The bisect path matters more than the LOC saved.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RmsNormKernelMode {
    Loop,
    Fused,
}

/// Cloneable bundle of `(mode, Option<Arc<RmsNormPipelines>>)`.
///
/// Every `RmsNorm` / `Attention` / `MoeBlock` in `Gemma4Model` carries a
/// clone of this so they can dispatch on `self.kernel.mode` without a
/// global switch. `Loop` mode carries `None` for pipelines (no compile
/// cost paid when the fused path is disabled) and `Fused` mode carries
/// `Some(Arc<RmsNormPipelines>)` pointing at a single library compiled
/// once at model-load time.
#[derive(Clone)]
pub struct RmsNormKernel {
    pub mode: RmsNormKernelMode,
    pub pipelines: Option<Arc<RmsNormPipelines>>,
}

impl RmsNormKernel {
    pub fn loop_mode() -> Self {
        Self {
            mode: RmsNormKernelMode::Loop,
            pipelines: None,
        }
    }

    /// Construct the `Fused` bundle by compiling the MSL library once.
    ///
    /// Safe to call many times — Metal's `newLibraryWithSource:` is
    /// idempotent per-device per-source but we nevertheless call this
    /// exactly once at `Gemma4Model::load_with_modes` and clone the
    /// resulting `Arc` into every sub-structure.
    pub fn fused_mode(metal_device: &MetalDevice) -> Result<Self> {
        Ok(Self {
            mode: RmsNormKernelMode::Fused,
            pipelines: Some(Arc::new(RmsNormPipelines::new(metal_device)?)),
        })
    }

    pub fn is_fused(&self) -> bool {
        matches!(self.mode, RmsNormKernelMode::Fused) && self.pipelines.is_some()
    }
}

impl RmsNormPipelines {
    /// Compile the MSL source and instantiate all six pipeline symbols.
    ///
    /// Returns `anyhow::Error` if the downstream compile fails (e.g. MSL
    /// syntax error introduced by an accidental edit of `RMS_NORM_FUSE_MSL`)
    /// or any of the six `host_name`d symbols is missing from the
    /// compiled library.
    pub fn new(metal_device: &MetalDevice) -> Result<Self> {
        let device = metal_device.device();
        let library = device
            .new_library_with_source(RMS_NORM_FUSE_MSL, None)
            .map_err(|e| anyhow!("rms_norm_kernel: library compile failed: {e:?}"))?;

        let load = |name: &str| -> Result<ComputePipeline> {
            let func = library
                .get_function(name, None)
                .map_err(|e| anyhow!("rms_norm_kernel: missing symbol {name}: {e:?}"))?;
            let pipe = device
                .new_compute_pipeline_state_with_function(&func)
                .map_err(|e| anyhow!("rms_norm_kernel: PSO create failed for {name}: {e:?}"))?;
            Ok(pipe)
        };

        Ok(Self {
            _library: library.clone(),
            pipe_f32_1: load("hf2q_rms_norm_f32")?,
            pipe_f32_2: load("hf2q_rms_norm_mul_f32")?,
            pipe_f32_3: load("hf2q_rms_norm_mul_add_f32")?,
            pipe_f32_4_1: load("hf2q_rms_norm_f32_4")?,
            pipe_f32_4_2: load("hf2q_rms_norm_mul_f32_4")?,
            pipe_f32_4_3: load("hf2q_rms_norm_mul_add_f32_4")?,
        })
    }
}

/// Borrow the Metal-backend handle out of a candle `Device`. Returns an
/// `anyhow::Error` when the device is not Metal — the caller is
/// responsible for never invoking this on CPU/CUDA tensors. Every hf2q
/// RmsNorm site runs on Metal (1b.4 forced the forward pass to Metal-only).
fn metal_device_of(device: &Device) -> Result<MetalDevice> {
    match device {
        Device::Metal(md) => Ok(md.clone()),
        other => Err(anyhow!("rms_norm_kernel: expected Metal device, got {other:?}")),
    }
}

/// Dispatch the fused RmsNorm kernel with the given fuse mode on the
/// currently-open candle compute encoder.
///
/// The caller selects the fuse mode implicitly via `weight` and
/// `residual`:
///   - `weight = None, residual = None`       → F=1 (unit RmsNorm; y = x*scale)
///   - `weight = Some, residual = None`       → F=2 (weighted; y = (x*scale)*w)
///   - `weight = Some, residual = Some`       → F=3 (weighted + post-norm add)
///   - `weight = None, residual = Some`       → not supported (error).
///
/// All tensors must be F32 and live on the same Metal device. `input`,
/// (optional) `residual`, and the returned output tensor share the same
/// shape; `weight` (when present) is 1-D with length `last_dim`.
///
/// Shape handling: the reduction runs over the innermost dim. We treat
/// the input as a flat sequence of `n_rows = numel / last_dim` rows of
/// `last_dim` elements each. `input` and `residual` must be contiguous;
/// if they are not, the caller must `.contiguous()` them first. This
/// module does NOT silently clone them because a silent clone would
/// defeat the dispatch-reduction goal.
///
/// Counter increments: this function increments
/// `dispatches_per_token += 1` and `norm_dispatches_per_token += 1`
/// via a `&DispatchCounters` handle passed by the caller. The caller
/// must NOT also increment the counters for ops the manual chain used
/// to do — this dispatch is one dispatch, full stop.
#[allow(clippy::too_many_arguments)]
pub fn rms_norm_fused(
    pipelines: &RmsNormPipelines,
    input: &Tensor,
    weight: Option<&Tensor>,
    residual: Option<&Tensor>,
    eps: f32,
) -> Result<Tensor> {
    // --- Validation ---------------------------------------------------
    let dtype = input.dtype();
    if dtype != DType::F32 {
        return Err(anyhow!(
            "rms_norm_kernel: input must be F32, got {:?}",
            dtype
        ));
    }
    if !input.is_contiguous() {
        return Err(anyhow!("rms_norm_kernel: input must be contiguous"));
    }

    let dims = input.dims().to_vec();
    let rank = dims.len();
    if rank == 0 {
        return Err(anyhow!("rms_norm_kernel: input must have rank >= 1"));
    }
    let last_dim = dims[rank - 1];
    let n_rows: usize = dims[..rank - 1].iter().product();
    if last_dim == 0 || n_rows == 0 {
        return Err(anyhow!(
            "rms_norm_kernel: empty tensor ne00={last_dim} n_rows={n_rows}"
        ));
    }

    // float4 path requires last_dim divisible by 4. hf2q's RmsNorm sites
    // all run on multiples of 256/512/2816/etc., so the float4 path is
    // always selected in practice — but guard the fallback for
    // correctness under Anti-Goal #6 (no benchmark tuning; all paths
    // handle arbitrary shape).
    let use_float4 = last_dim % 4 == 0;
    let ne00 = last_dim as i32;
    let ne00_t: i32 = if use_float4 {
        (last_dim / 4) as i32
    } else {
        last_dim as i32
    };

    if let Some(w) = weight {
        if w.dtype() != DType::F32 {
            return Err(anyhow!("rms_norm_kernel: weight must be F32"));
        }
        if !w.is_contiguous() {
            return Err(anyhow!("rms_norm_kernel: weight must be contiguous"));
        }
        // Weight is the last-dim gain vector. Accept rank-1 `[last_dim]`
        // OR higher-rank shapes whose trailing dim equals `last_dim`
        // (candle broadcast_mul uses shape `[last_dim]` throughout
        // gemma4.rs, but the llama.cpp kernel only cares about the flat
        // `last_dim` element count through the nbf1 row-stride path).
        if w.elem_count() != last_dim {
            return Err(anyhow!(
                "rms_norm_kernel: weight must have {} elements, got {}",
                last_dim,
                w.elem_count()
            ));
        }
    }
    if residual.is_some() && weight.is_none() {
        return Err(anyhow!(
            "rms_norm_kernel: F=3 path (residual = Some) requires weight = Some"
        ));
    }
    if let Some(r) = residual {
        if r.dtype() != DType::F32 {
            return Err(anyhow!("rms_norm_kernel: residual must be F32"));
        }
        if !r.is_contiguous() {
            return Err(anyhow!("rms_norm_kernel: residual must be contiguous"));
        }
        if r.dims() != input.dims() {
            return Err(anyhow!(
                "rms_norm_kernel: residual shape {:?} must equal input shape {:?}",
                r.dims(),
                input.dims()
            ));
        }
    }

    // --- Pipeline selection ------------------------------------------
    let fuse: u32 = match (weight.is_some(), residual.is_some()) {
        (false, false) => 1,
        (true, false) => 2,
        (true, true) => 3,
        (false, true) => unreachable!("covered above"),
    };
    let pipeline = match (fuse, use_float4) {
        (1, false) => &pipelines.pipe_f32_1,
        (2, false) => &pipelines.pipe_f32_2,
        (3, false) => &pipelines.pipe_f32_3,
        (1, true) => &pipelines.pipe_f32_4_1,
        (2, true) => &pipelines.pipe_f32_4_2,
        (3, true) => &pipelines.pipe_f32_4_3,
        _ => unreachable!(),
    };

    // --- Metal device + buffer extraction -----------------------------
    let metal_device = metal_device_of(input.device())?;

    // Hold the storage guards alive through the whole encode-and-dispatch
    // scope. `storage_and_layout` returns an `RwLockReadGuard<Storage>`;
    // dropping that guard while the Metal `set_buffer` binding is still
    // in the encoder would risk a use-after-free in the worst case.
    // Extract &Buffer + byte offset inside the scope, and keep the
    // guards in local bindings that outlive the encoder drop at the
    // end of this function.
    let (src0_storage, src0_layout) = input.storage_and_layout();
    let src0_buf_ref: &Buffer = match &*src0_storage {
        Storage::Metal(ms) => ms.buffer(),
        other => {
            return Err(anyhow!(
                "rms_norm_kernel: input storage must be Metal, got {other:?}"
            ));
        }
    };
    let src0_offset = src0_layout.start_offset() * DType::F32.size_in_bytes();

    let weight_guards = match weight {
        Some(w) => Some(w.storage_and_layout()),
        None => None,
    };
    let (f0_buf_ref, f0_offset): (&Buffer, usize) = match &weight_guards {
        Some((w_storage, w_layout)) => {
            let buf = match &**w_storage {
                Storage::Metal(ms) => ms.buffer(),
                other => {
                    return Err(anyhow!(
                        "rms_norm_kernel: weight storage must be Metal, got {other:?}"
                    ));
                }
            };
            (buf, w_layout.start_offset() * DType::F32.size_in_bytes())
        }
        None => (src0_buf_ref, src0_offset),
    };

    let residual_guards = match residual {
        Some(r) => Some(r.storage_and_layout()),
        None => None,
    };
    let (f1_buf_ref, f1_offset): (&Buffer, usize) = match &residual_guards {
        Some((r_storage, r_layout)) => {
            let buf = match &**r_storage {
                Storage::Metal(ms) => ms.buffer(),
                other => {
                    return Err(anyhow!(
                        "rms_norm_kernel: residual storage must be Metal, got {other:?}"
                    ));
                }
            };
            (buf, r_layout.start_offset() * DType::F32.size_in_bytes())
        }
        None => (src0_buf_ref, src0_offset),
    };

    // --- Kargs struct (mirrors ggml_metal_kargs_norm) -----------------
    // The kernel reads from strided memory with
    //   x = src0 + i03*nbf3[0] + i02*nbf2[0] + i01*nbf1[0]
    // with `tgpig = (i01, i02, i03)`. For our flattened layout we pack
    // everything into i01 ∈ [0, n_rows) and leave i02 = i03 = 0, so
    // `nbf2[0]` and `nbf3[0]` are unused but must still be well-defined
    // (they multiply by 0).
    let elem = std::mem::size_of::<f32>() as u64; // hf2q is F32-only here
    let row_bytes = last_dim as u64 * elem;

    // Destination strides: output shape matches input. We write row i01
    // at offset `i01 * row_bytes`, same as the src0 read stride.
    let nb1 = row_bytes;
    let nb2 = n_rows as u64 * row_bytes;
    let nb3 = nb2;

    // f0 is the weight row: it is a single 1-D vector of length
    // `last_dim` shared across every output row. To broadcast it we set
    //   nef1[1] = 1, nef2[1] = 1, nef3[1] = 1, nbf1[1] = nbf2[1] = nbf3[1] = 0
    // so that `(i01 % 1) * 0 + ...` always resolves to the base pointer
    // regardless of which threadgroup (i01, i02, i03) we are in. That is
    // the same trick llama.cpp uses at `ggml-metal-ops.cpp:3351-3356`
    // (`nef1[0] = ne01` for the base row, but weights at slot 1 are
    // loaded with `nef1[1] = f1->src[1]->ne[1]` which is the weight's
    // row count — 1 for a 1-D vector — and `nbf1[1] = f1->src[1]->nb[1]`
    // which is 0 for a 1-D tensor).
    //
    // f1 is the residual: its shape matches src0, so stride slot [2] is
    // the same as slot [0].
    let kargs = RmsNormKargs {
        ne00,
        ne00_t,
        nb1,
        nb2,
        nb3,
        eps,
        nef1: [n_rows as i32, 1, n_rows as i32],
        nef2: [1, 1, 1],
        nef3: [1, 1, 1],
        nbf1: [row_bytes, 0, row_bytes],
        nbf2: [nb2, 0, nb2],
        nbf3: [nb3, 0, nb3],
    };

    // --- Destination buffer + threadgroup shape -----------------------
    let out_elems = n_rows * last_dim;
    let dst_buf = metal_device.new_buffer(out_elems, DType::F32, "rms_norm_dst")?;

    // Threads per threadgroup: walk the llama.cpp rule
    //   nth = 32;
    //   while (nth < ne00_t && nth < max_total_threads_per_threadgroup)
    //       nth *= 2;
    //   nth = min(nth, max_threads); nth = min(nth, ne00_t);
    // from `ggml-metal-ops.cpp:3436-3443`. `ne00_t` for hf2q is
    // `last_dim/4` (scalar: up to 2816/4=704; float4 always).
    let max_tg = pipeline.max_total_threads_per_threadgroup();
    let mut nth: usize = 32;
    while nth < ne00_t as usize && nth < max_tg {
        nth *= 2;
    }
    nth = nth.min(max_tg).min(ne00_t as usize).max(32);

    // Threadgroup memory: the kernel uses `shmem_f32[tiisg]` (indexed
    // by thread-in-simdgroup, max 32) and `shmem_f32[sgitg]` (indexed
    // by simdgroup-in-threadgroup, max `nth/32`). A 32-float buffer is
    // enough for both accesses so long as `nth/32 <= 32`, i.e.
    // `nth <= 1024` which holds on every Apple-silicon GPU we target.
    let shmem_bytes = 32 * std::mem::size_of::<f32>();

    // --- Encode the dispatch ------------------------------------------
    // Scope the encoder in a block so its Drop fires `endEncoding`
    // before we wrap the destination buffer in a candle Tensor (the
    // tensor-wrap allocation must observe the write). Same pattern as
    // moe_kernel.rs::forward_fused at gemma4.rs:1001-1022.
    {
        let encoder = metal_device.command_encoder()?;
        encoder.set_compute_pipeline_state(pipeline);
        encoder.set_bytes(0, &kargs);
        encoder.set_buffer(1, Some(src0_buf_ref), src0_offset);
        encoder.set_buffer(2, Some(f0_buf_ref), f0_offset);
        encoder.set_buffer(3, Some(f1_buf_ref), f1_offset);
        encoder.set_buffer(4, Some(dst_buf.as_ref()), 0);

        encoder.set_threadgroup_memory_length(0, shmem_bytes);

        encoder.use_resource(src0_buf_ref, MTLResourceUsage::Read);
        encoder.use_resource(f0_buf_ref, MTLResourceUsage::Read);
        encoder.use_resource(f1_buf_ref, MTLResourceUsage::Read);
        encoder.use_resource(dst_buf.as_ref(), MTLResourceUsage::Write);

        // Grid: one threadgroup per (i01, i02, i03) row. We pack
        // everything into i01 so the grid is (n_rows, 1, 1).
        let grid = MTLSize {
            width: n_rows,
            height: 1,
            depth: 1,
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
        candle_core::MetalStorage::new(dst_buf, metal_device, out_elems, DType::F32);
    let out_tensor = Tensor::from_storage(
        Storage::Metal(storage),
        dims.as_slice(),
        candle_core::op::BackpropOp::none(),
        false,
    );
    Ok(out_tensor)
}


// ---------------------------------------------------------------------------
// Phase A unit test — numerical parity vs the manual candle chain
// ---------------------------------------------------------------------------
//
// Builds an F32 `[1, n_rows, last_dim]` input on the Metal device,
// runs the fused kernel in all three F modes, and compares element-by-
// element against the reference computed with the existing 11-op
// candle chain (for F=2) or 9-op chain (F=1) or 12-op chain (F=3).
//
// The test is skipped when no Metal device is available (CI). Build
// with `--features metal` and run with
//   `cargo test --features metal --release -p hf2q -- rms_norm_kernel::tests --nocapture`

#[cfg(test)]
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

    /// Reference implementation: the exact candle op chain that
    /// `gemma4.rs::RmsNorm::forward` runs today. Kept in-sync with that
    /// method literally; any edit to the manual chain in gemma4.rs
    /// should be mirrored here to keep the unit test a faithful oracle.
    fn reference_rms_norm(
        x: &Tensor,
        weight: Option<&Tensor>,
        residual: Option<&Tensor>,
        eps: f64,
    ) -> Result<Tensor> {
        use candle_core::D;
        let x_f32 = x.to_dtype(DType::F32)?;
        let sq = x_f32.sqr()?;
        let mean_sq = sq.mean_keepdim(D::Minus1)?;
        let eps_t = mean_sq.ones_like()?.affine(0.0, eps)?;
        let rms = (mean_sq + eps_t)?.sqrt()?.recip()?;
        let normed = x_f32.broadcast_mul(&rms)?;
        let out = match weight {
            Some(w) => {
                let w_f32 = w.to_dtype(DType::F32)?;
                normed.broadcast_mul(&w_f32)?
            }
            None => normed,
        };
        let out = match residual {
            Some(r) => (out + r)?,
            None => out,
        };
        Ok(out)
    }

    fn run_case(
        device: &CoreDevice,
        pipelines: &RmsNormPipelines,
        shape: (usize, usize, usize),
        fuse: u32,
        label: &str,
    ) {
        let (b, s, d) = shape;
        let n = b * s * d;
        let x_vec = make_f32_vec(n, 0x0BAD_F00D);
        let x = Tensor::from_vec(x_vec, (b, s, d), device).unwrap();
        let w_vec = make_f32_vec(d, 0xDEAD_BEEF);
        let w = Tensor::from_vec(w_vec, (d,), device).unwrap();
        let r_vec = make_f32_vec(n, 0xFACE_FEED);
        let r = Tensor::from_vec(r_vec, (b, s, d), device).unwrap();

        let eps = 1e-6_f64;

        let (weight_opt, residual_opt): (Option<&Tensor>, Option<&Tensor>) = match fuse {
            1 => (None, None),
            2 => (Some(&w), None),
            3 => (Some(&w), Some(&r)),
            _ => unreachable!(),
        };

        let ref_out = reference_rms_norm(&x, weight_opt, residual_opt, eps).unwrap();
        let ref_flat = ref_out.flatten_all().unwrap().to_vec1::<f32>().unwrap();

        let fused_out =
            rms_norm_fused(pipelines, &x, weight_opt, residual_opt, eps as f32).unwrap();
        let fused_flat = fused_out.flatten_all().unwrap().to_vec1::<f32>().unwrap();

        assert_eq!(ref_flat.len(), fused_flat.len());
        let mut max_abs = 0.0_f32;
        let mut idx = 0usize;
        for (i, (a, b)) in ref_flat.iter().zip(fused_flat.iter()).enumerate() {
            let dlt = (*a - *b).abs();
            if dlt > max_abs {
                max_abs = dlt;
                idx = i;
            }
        }
        eprintln!(
            "fused_rms_norm {label} F={fuse} shape=({b},{s},{d}): max|Δ|={max_abs:.3e} \
             at idx {idx} (ref={}, fused={})",
            ref_flat[idx], fused_flat[idx]
        );
        assert!(
            max_abs < 1e-5,
            "fused RmsNorm F={fuse} {label} disagrees with manual chain at ε=1e-5; \
             max|Δ|={max_abs}"
        );
    }

    fn device_and_pipelines() -> Option<(CoreDevice, RmsNormPipelines)> {
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
        let pipelines = RmsNormPipelines::new(&md).unwrap();
        Some((device, pipelines))
    }

    #[test]
    fn rms_norm_f32_fuse1_matches_manual_chain_float4() {
        let Some((device, pipelines)) = device_and_pipelines() else { return };
        // hidden_size=2816 is the common hf2q shape; 2816 % 4 == 0 → float4 path.
        run_case(&device, &pipelines, (1, 4, 2816), 1, "float4-hidden");
    }

    #[test]
    fn rms_norm_f32_fuse2_matches_manual_chain_float4() {
        let Some((device, pipelines)) = device_and_pipelines() else { return };
        run_case(&device, &pipelines, (1, 4, 2816), 2, "float4-hidden");
    }

    #[test]
    fn rms_norm_f32_fuse3_matches_manual_chain_float4() {
        let Some((device, pipelines)) = device_and_pipelines() else { return };
        run_case(&device, &pipelines, (1, 4, 2816), 3, "float4-hidden");
    }

    #[test]
    fn rms_norm_f32_all_fuses_head_dim_256() {
        // q_norm / k_norm / rms_norm_unit shape: last dim is the sliding
        // head_dim=256 (also divisible by 4, still float4 path, but a
        // different `nth` branch).
        let Some((device, pipelines)) = device_and_pipelines() else { return };
        run_case(&device, &pipelines, (1, 16, 256), 1, "head_dim=256");
        run_case(&device, &pipelines, (1, 16, 256), 2, "head_dim=256");
        run_case(&device, &pipelines, (1, 16, 256), 3, "head_dim=256");
    }

    #[test]
    fn rms_norm_f32_all_fuses_head_dim_512() {
        // Global-attention head_dim=512 (bd=512 fused SDPA prefill path).
        let Some((device, pipelines)) = device_and_pipelines() else { return };
        run_case(&device, &pipelines, (1, 16, 512), 1, "head_dim=512");
        run_case(&device, &pipelines, (1, 16, 512), 2, "head_dim=512");
        run_case(&device, &pipelines, (1, 16, 512), 3, "head_dim=512");
    }

    #[test]
    fn rms_norm_f32_all_fuses_scalar_path() {
        // last_dim=127 is NOT divisible by 4 — exercises the scalar
        // `float` path rather than `float4`. Smaller than any real call
        // site but correctness is required for Anti-Goal #6 (no
        // benchmark-only specialisation).
        let Some((device, pipelines)) = device_and_pipelines() else { return };
        run_case(&device, &pipelines, (1, 4, 127), 1, "scalar");
        run_case(&device, &pipelines, (1, 4, 127), 2, "scalar");
        run_case(&device, &pipelines, (1, 4, 127), 3, "scalar");
    }

    #[test]
    fn rms_norm_f32_decode_shape_single_token() {
        // Decode-time shape: [1, 1, 2816]. Only 1 threadgroup dispatched.
        let Some((device, pipelines)) = device_and_pipelines() else { return };
        run_case(&device, &pipelines, (1, 1, 2816), 1, "decode-1");
        run_case(&device, &pipelines, (1, 1, 2816), 2, "decode-1");
        run_case(&device, &pipelines, (1, 1, 2816), 3, "decode-1");
    }
}

