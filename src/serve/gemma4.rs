//! Gemma 4 A4B model — decoder-only transformer with Mixture of Experts.
//!
//! Architecture: attention + dense MLP + SigMoE (128 experts, top-8) per layer.
//! Dual attention: sliding (head_dim=256) and global (head_dim=512) layers.
//! RoPE: standard for sliding, partial for global.

use anyhow::Result;
use candle_core::{DType, Device, Tensor, D};
use candle_core::quantized::QMatMul;
use candle_nn::{Embedding, Module};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use super::config::Gemma4Config;
use super::gguf_loader::GgufModel;
use super::lm_head_kernel::{self, LmHeadKernelMode as LmHeadKernelModeImpl};
use super::moe_kernel;
use super::rms_norm_kernel::{self, RmsNormKernel, RmsNormKernelMode as RmsKernelMode};
use super::rope_kernel::{self, RopeKernel, RopeKernelMode as RopeKernelModeImpl, RopeVariant};

// ADR-005 1bNEW.1 — type alias for the quantized dtype tag used by
// `call_quantized_matmul_mv_id_t`. Kept in sync with candle-core's
// `quantized::GgmlDType` at `QTensor::dtype()` return time.
#[cfg(feature = "metal")]
use candle_metal_kernels::GgmlDType as MetalGgmlDType;

/// Per-model switch for the MoE expert dispatch path. Populated from the
/// CLI `--moe-kernel` flag (see `cli::MoeKernelMode`) at `Gemma4Model::load`
/// time and propagated down to each `MoeBlock` based on the layer index.
///
/// Phase B (ADR-005 1bNEW.1, 2026-04-10): `Fused` activates the fused
/// `kernel_mul_mv_id_*` path for layer 0 ONLY; every other layer keeps
/// the `Loop` baseline. Phase C will widen `Fused` to all layers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MoeKernelMode {
    Loop,
    Fused,
}

impl From<crate::cli::MoeKernelMode> for MoeKernelMode {
    fn from(v: crate::cli::MoeKernelMode) -> Self {
        match v {
            crate::cli::MoeKernelMode::Loop => Self::Loop,
            crate::cli::MoeKernelMode::Fused => Self::Fused,
        }
    }
}

impl From<crate::cli::RmsNormKernelMode> for RmsKernelMode {
    fn from(v: crate::cli::RmsNormKernelMode) -> Self {
        match v {
            crate::cli::RmsNormKernelMode::Loop => Self::Loop,
            crate::cli::RmsNormKernelMode::Fused => Self::Fused,
        }
    }
}

impl From<crate::cli::RopeKernelMode> for RopeKernelModeImpl {
    fn from(v: crate::cli::RopeKernelMode) -> Self {
        match v {
            crate::cli::RopeKernelMode::Loop => Self::Loop,
            crate::cli::RopeKernelMode::Fused => Self::Fused,
        }
    }
}

/// ADR-005 1bNEW.20 — KV cache append dispatch mode.
///
/// `SliceScatter` preserves the pre-1bNEW.20 path: two `Tensor::slice_scatter`
/// writes into the pre-allocated cache followed by `narrow` + `contiguous`
/// on the active region. The `contiguous()` exists specifically to work
/// around the slice_scatter dim-!=-0 transpose-trick stride gotcha that
/// produced the a0952e2 silent-gibberish regression — see the comment block
/// inside `KvCache::append` below for the full history.
///
/// `InPlace` is the Walk-KERNEL-PORT of llama.cpp's `llama_kv_cache::cpy_k`
/// / `cpy_v` pattern at `/opt/llama.cpp/src/llama-kv-cache.cpp:1196-1285`.
/// llama.cpp uses `ggml_set_rows` / `ggml_cpy` to write the new K/V slots
/// directly into a view of the pre-allocated cache at a computed offset,
/// without any copy of the active region on read. hf2q's equivalent is
/// candle's `Tensor::slice_set` at `candle-core/src/tensor_cat.rs:246`,
/// which performs an in-place `storage.copy2d` from `src` into `self`'s
/// Metal buffer at `offset * block_size` in elements — no `slice_scatter`,
/// no transpose, no contiguous copy. The returned active-region view is
/// a plain `narrow` on dim 2 of the pre-allocated cache buffer; because
/// the cache was zero-allocated as `[1, kv_heads, cache_size, hd]`
/// contiguous and only ever written to in-place, its per-head stride
/// (`k_stride[1] = cache_size * hd`) is the same as it would be for a
/// full-cache view. Candle's SDPA vector kernel at
/// `candle-metal-kernels/src/kernels/sdpa.rs:278-279` reads this stride
/// explicitly and walks only the first `visible_len` positions per head,
/// so the narrow-without-contiguous view is stride-correct for decode.
///
/// Prefill paths that must consume a contiguous layout (global bd=512
/// path's `.to_dtype(BF16)?.contiguous()?` at
/// `src/serve/gemma4.rs:~843` and sliding manual path's `reshape` at
/// `src/serve/gemma4.rs:~897-908`) add their own `.contiguous()` on the
/// view — the same copy the `SliceScatter` path used to pay inside
/// `append`, but now only once per prefill pass instead of on every
/// decode step.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KvCacheKernelMode {
    SliceScatter,
    InPlace,
}

impl From<crate::cli::KvCacheKernelMode> for KvCacheKernelMode {
    fn from(v: crate::cli::KvCacheKernelMode) -> Self {
        match v {
            crate::cli::KvCacheKernelMode::SliceScatter => Self::SliceScatter,
            crate::cli::KvCacheKernelMode::InPlace => Self::InPlace,
        }
    }
}

const MODEL_DTYPE: DType = DType::F32;

// ---------------------------------------------------------------------------
// DispatchCounters — ADR-005 1bNEW.0 (pre-flight metrics instrumentation)
// ---------------------------------------------------------------------------
//
// Observe-only per-forward-pass counters used by the Walk progress discipline
// (ADR-005 line 157, stop-and-diagnose rule). Each counter is an AtomicU64
// incremented at the corresponding dispatch-issuance site in the forward path.
//
// Semantics:
//   - `dispatches_per_token`      : total candle/Metal op calls issued by one
//                                    `Gemma4Model::forward` pass. Coarse — the
//                                    sum of every other counter plus the ops
//                                    that don't fit another bucket (embedding
//                                    lookup, final matmul, softcapping).
//   - `moe_to_vec2_count`         : `Tensor::to_vec2()` calls inside
//                                    `MoeBlock::forward` (gemma4.rs:428-429).
//                                    Each is a forced `waitUntilCompleted`.
//                                    Baseline expectation: 2 × num_layers.
//   - `moe_dispatches`            : total candle ops inside every MoeBlock
//                                    forward across all layers.
//   - `moe_layer_invocations`     : number of MoeBlock forward calls this
//                                    pass (used to average moe_dispatches).
//   - `sampler_sync_count`        : `argmax().to_scalar()` forced GPU→CPU
//                                    syncs in the sampler (sampler.rs:49).
//                                    Baseline: 1 per decoded token.
//   - `norm_dispatches_per_token` : total candle ops inside RmsNorm::forward
//                                    and Attention::rms_norm_unit (norm-path
//                                    dispatches).
//   - `forward_count`             : number of `Gemma4Model::forward` calls
//                                    since the last reset. Used as the divisor
//                                    when computing per-token averages.
//
// All increments are `Ordering::Relaxed` — we only need atomicity, not a
// happens-before relation, because the counters are never read during a
// forward pass, only after the benchmark loop completes.
//
// Chesterton's fence: no counters exist today. Phase 1 shipped with a single
// `Instant::now()` measurement at the `generate()` boundary in mod.rs; this
// struct is pure visibility, not a replacement for a hidden mechanism.
#[derive(Default)]
pub struct DispatchCounters {
    pub dispatches_per_token: AtomicU64,
    pub moe_to_vec2_count: AtomicU64,
    pub moe_dispatches: AtomicU64,
    pub moe_layer_invocations: AtomicU64,
    pub sampler_sync_count: AtomicU64,
    pub norm_dispatches_per_token: AtomicU64,
    pub forward_count: AtomicU64,
}

impl DispatchCounters {
    pub fn new() -> Arc<Self> {
        Arc::new(Self::default())
    }

    /// Zero every field. Called once before the decode loop so that counters
    /// reflect only the timed region (decode), not prefill or warmup.
    pub fn reset(&self) {
        self.dispatches_per_token.store(0, Ordering::Relaxed);
        self.moe_to_vec2_count.store(0, Ordering::Relaxed);
        self.moe_dispatches.store(0, Ordering::Relaxed);
        self.moe_layer_invocations.store(0, Ordering::Relaxed);
        self.sampler_sync_count.store(0, Ordering::Relaxed);
        self.norm_dispatches_per_token.store(0, Ordering::Relaxed);
        self.forward_count.store(0, Ordering::Relaxed);
    }

    /// Convenience snapshot of the current values.
    pub fn snapshot(&self) -> DispatchSnapshot {
        DispatchSnapshot {
            dispatches_per_token: self.dispatches_per_token.load(Ordering::Relaxed),
            moe_to_vec2_count: self.moe_to_vec2_count.load(Ordering::Relaxed),
            moe_dispatches: self.moe_dispatches.load(Ordering::Relaxed),
            moe_layer_invocations: self.moe_layer_invocations.load(Ordering::Relaxed),
            sampler_sync_count: self.sampler_sync_count.load(Ordering::Relaxed),
            norm_dispatches_per_token: self.norm_dispatches_per_token.load(Ordering::Relaxed),
            forward_count: self.forward_count.load(Ordering::Relaxed),
        }
    }
}

/// Plain-value snapshot of counters. Used for averaging and report emission.
#[derive(Debug, Clone, Copy, Default)]
pub struct DispatchSnapshot {
    pub dispatches_per_token: u64,
    pub moe_to_vec2_count: u64,
    pub moe_dispatches: u64,
    pub moe_layer_invocations: u64,
    pub sampler_sync_count: u64,
    pub norm_dispatches_per_token: u64,
    pub forward_count: u64,
}

impl DispatchSnapshot {
    /// Per-forward-pass (= per decoded token) average. Division by zero is
    /// guarded with `forward_count.max(1)` so an un-reset run still reports
    /// well-formed numbers.
    pub fn per_token(&self) -> PerTokenMetrics {
        let n = self.forward_count.max(1) as f64;
        let moe_layers = self.moe_layer_invocations.max(1) as f64;
        PerTokenMetrics {
            dispatches_per_token: self.dispatches_per_token as f64 / n,
            moe_to_vec2_count: self.moe_to_vec2_count as f64 / n,
            moe_dispatches_per_layer: self.moe_dispatches as f64 / moe_layers,
            sampler_sync_count: self.sampler_sync_count as f64 / n,
            norm_dispatches_per_token: self.norm_dispatches_per_token as f64 / n,
        }
    }
}

/// Per-token averages (the numbers written to `metrics.txt`).
#[derive(Debug, Clone, Copy, Default)]
pub struct PerTokenMetrics {
    pub dispatches_per_token: f64,
    pub moe_to_vec2_count: f64,
    pub moe_dispatches_per_layer: f64,
    pub sampler_sync_count: f64,
    pub norm_dispatches_per_token: f64,
}

// ---------------------------------------------------------------------------
// RmsNorm
// ---------------------------------------------------------------------------

struct RmsNorm {
    weight: Tensor,
    eps: f64,
    counters: Arc<DispatchCounters>,
    /// ADR-005 1bNEW.4: per-site dispatch mode plumbed from the CLI
    /// `--rms-norm-kernel` flag. `Fused` carries an `Arc<RmsNormPipelines>`;
    /// `Loop` carries `None`.
    kernel: RmsNormKernel,
}

impl RmsNorm {
    fn new(
        weight: Tensor,
        eps: f64,
        counters: Arc<DispatchCounters>,
        kernel: RmsNormKernel,
    ) -> Self {
        Self { weight, eps, counters, kernel }
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        if self.kernel.is_fused() {
            // ADR-005 1bNEW.4 — single dispatch replaces the 11-op chain.
            // Caller must hand in an F32 contiguous input; every hf2q
            // RmsNorm site already meets this invariant post-1b.4.
            let pipelines = self.kernel.pipelines.as_ref().expect("is_fused");
            let x_c = if x.is_contiguous() { x.clone() } else { x.contiguous()? };
            let out = rms_norm_kernel::rms_norm_fused(
                pipelines,
                &x_c,
                Some(&self.weight),
                None,
                self.eps as f32,
            )?;
            self.counters.norm_dispatches_per_token.fetch_add(1, Ordering::Relaxed);
            self.counters.dispatches_per_token.fetch_add(1, Ordering::Relaxed);
            return Ok(out);
        }

        let dtype = x.dtype();
        let x_f32 = x.to_dtype(DType::F32)?;
        let sq = x_f32.sqr()?;
        let mean_sq = sq.mean_keepdim(D::Minus1)?;
        let eps_t = mean_sq.ones_like()?.affine(0.0, self.eps)?;
        let rms = (mean_sq + eps_t)?.sqrt()?.recip()?;
        let normed = x_f32.broadcast_mul(&rms)?;
        let weight_f32 = self.weight.to_dtype(DType::F32)?;
        let result = normed.broadcast_mul(&weight_f32)?;
        let out = result.to_dtype(dtype)?;
        // 11 candle ops above: to_dtype, sqr, mean_keepdim, ones_like+affine (1),
        // (mean_sq + eps_t) (1), sqrt, recip, broadcast_mul, to_dtype(weight),
        // broadcast_mul, to_dtype(out). Counted as 11 dispatches; the trailing
        // to_dtype is a no-op at F32 steady state but still counts as a call
        // into candle (matches ADR-005 line 284 accounting).
        self.counters.norm_dispatches_per_token.fetch_add(11, Ordering::Relaxed);
        self.counters.dispatches_per_token.fetch_add(11, Ordering::Relaxed);
        Ok(out)
    }

    /// ADR-005 1bNEW.4 F=3 fused path: norm(x) then add residual, in a
    /// single dispatch when the fused kernel is wired. Used by the one
    /// NORM→ADD site in `DecoderLayer::forward` at the post-FFW
    /// combiner. In `Loop` mode this degenerates to the explicit
    /// two-op pattern `xs = residual + forward(x)` that 1bNEW.0b
    /// un-fused from the old `forward_with_residual` — preserving
    /// that reference-citable lay-out when the fused path is off.
    ///
    /// Walk Exception Register note: 1bNEW.0b un-fused the
    /// ADD-THEN-NORM pattern at the *pre-FFW* site (which still stays
    /// un-fused — see `DecoderLayer::forward`). This method handles
    /// the NORM-THEN-ADD pattern at the *post-FFW combiner* site
    /// (gemma4.rs:1228-1229 pre-1bNEW.4), which is a DIFFERENT
    /// operation and IS present in both mlx-lm and llama.cpp
    /// references as NORM-first. The fused F=3 kernel computes exactly
    /// `(x*scale)*w + residual`, matching the reference order
    /// byte-for-byte.
    fn forward_with_post_residual(
        &self,
        x: &Tensor,
        residual: &Tensor,
    ) -> Result<Tensor> {
        if self.kernel.is_fused() {
            let pipelines = self.kernel.pipelines.as_ref().expect("is_fused");
            let x_c = if x.is_contiguous() { x.clone() } else { x.contiguous()? };
            let r_c = if residual.is_contiguous() {
                residual.clone()
            } else {
                residual.contiguous()?
            };
            let out = rms_norm_kernel::rms_norm_fused(
                pipelines,
                &x_c,
                Some(&self.weight),
                Some(&r_c),
                self.eps as f32,
            )?;
            self.counters.norm_dispatches_per_token.fetch_add(1, Ordering::Relaxed);
            self.counters.dispatches_per_token.fetch_add(1, Ordering::Relaxed);
            return Ok(out);
        }

        // Loop mode: explicit NORM-THEN-ADD two-op pattern, same as the
        // inlined pattern at the pre-1bNEW.4 call site. Counted ops:
        // the forward() call self-counts 11; the add is 1.
        let normed = self.forward(x)?;
        let out = (residual + normed)?;
        self.counters.dispatches_per_token.fetch_add(1, Ordering::Relaxed);
        Ok(out)
    }

    // ADR-005 1bNEW.0b: the previous `forward_with_residual` helper (Phase 1
    // 1b.10, lines 47-51 of `0a703d7`) has been removed. References unfuse
    // the post-attention residual add from the pre-FFW norm; see the inline
    // call site in `DecoderLayer::forward` for citations.
}

// ---------------------------------------------------------------------------
// Rotary Embedding
// ---------------------------------------------------------------------------

struct RotaryEmbedding {
    /// Cached sin table `[max_seq_len, head_dim/2]` — only used by
    /// the `loop` dispatch path. `Fused` mode computes angles
    /// on-the-fly inside the kernel body, so this allocation is
    /// pure fallback cost when `--rope-kernel=fused` is active.
    /// Left in place unconditionally for bisect-safety.
    ///
    /// **ADR-005 1bNEW.18 (2026-04-11):** for global layers this
    /// table is built with each per-pair `inv_freq[i]` pre-divided
    /// by `freq_factors[i]` from `rope_freqs.weight` (the F32 `[256]`
    /// tensor llama.cpp calls `model.layers[il].rope_freqs` at
    /// `/opt/llama.cpp/src/models/gemma4-iswa.cpp:55-59,73-75,97-98`).
    /// Pair indices with `freq_factors[i] == 1e+30` produce
    /// `freqs[pos, i] ≈ 0 → cos=1, sin=0 → identity rotation`, which
    /// is numerically equivalent to what the Metal kernel computes
    /// via `theta / freq_factors[ic]` at runtime
    /// (`/opt/llama.cpp/ggml/src/ggml-metal/ggml-metal.metal:4353`).
    sin: Tensor,
    /// Cached cos table, same shape and caveat as `sin`.
    cos: Tensor,
    /// Full head_dim this RoPE is paired with. `Loop` mode splits
    /// `q[..head_dim]` in half at offset `head_dim/2`; fused mode
    /// passes `head_dim` as both `ne0` and `n_dims` so llama.cpp's
    /// kernel pairs `(q[ic], q[ic + head_dim/2])` — the canonical
    /// HF neox layout. Note: pre-1bNEW.18, there was a separate
    /// `rotary_dim` field that could differ from `head_dim` under
    /// a misreading of Gemma 4's "partial rotary"; the field was
    /// deleted and the two-value-distinction collapsed to `head_dim`
    /// only. See `docs/spike-C-results.md` Part 5 for the root cause.
    head_dim: usize,
    /// RoPE base frequency as read from GGUF metadata
    /// (`gemma4.rope.freq_base` = 1e6 for global,
    /// `gemma4.rope.freq_base_swa` = 10000 for sliding). The fused
    /// kernel path passes this verbatim as `freq_base` to the ported
    /// `kernel_rope_neox`, which computes
    /// `theta = pos * pow(freq_base, -i0/n_dims) / freq_factor`.
    rope_theta: f32,
    /// Global-layer frequency mask loaded from `rope_freqs.weight`.
    /// Shape `[head_dim/2]` F32 on device, populated on global
    /// `RotaryEmbedding` instances only; `None` on sliding instances.
    /// When present, bound as `src2` to the fused kernel so the
    /// `args.src2 != 0` branch at `rope_kernel.rs:319` activates.
    /// Reference: `/opt/llama.cpp/src/models/gemma4-iswa.cpp:55-59`
    /// — `freq_factors = model.layers[il].rope_freqs` on non-SWA
    /// layers only.
    freq_factors: Option<Tensor>,
    /// Which RoPE pair layout this instance dispatches. Gemma 4 is
    /// always `Neox` (llama.cpp's split-half `LLAMA_ROPE_TYPE_NEOX`
    /// at `src/llama-model.cpp:9134`). Stored as a struct field so
    /// future model families using GPT-J interleaved rotation can
    /// share the same pipelines via `RopeVariant::Norm`.
    variant: RopeVariant,
    /// Per-site dispatch mode plumbed from the CLI `--rope-kernel`
    /// flag. `Fused` carries an `Arc<RopePipelines>`; `Loop` carries
    /// `None` — mirrors `RmsNormKernel` from 1bNEW.4.
    kernel: RopeKernel,
    /// Shared counter Arc. `Loop` mode increments the old 9-or-10-
    /// op count; `Fused` mode increments `+1` per dispatch to match
    /// 1bNEW.1 / 1bNEW.4 accounting discipline.
    counters: Arc<DispatchCounters>,
}

impl RotaryEmbedding {
    /// Construct a RoPE embedding with full-head rotation.
    ///
    /// **Sliding layers:** `freq_factors = None`, which is numerically
    /// equivalent to passing `[1.0; head_dim/2]` — every pair
    /// rotates at its natural per-index angle.
    ///
    /// **Global layers (Gemma 4 full-attention):** `freq_factors =
    /// Some(rope_freqs_host)` where `rope_freqs_host` is the F32
    /// vector loaded from `rope_freqs.weight` in the GGUF. Pair
    /// indices with a `1e+30` mask value effectively have their
    /// rotation angle driven to 0 (identity rotation), matching
    /// llama.cpp's `theta / freq_factors[ic]` runtime division at
    /// `/opt/llama.cpp/ggml/src/ggml-metal/ggml-metal.metal:4353`.
    ///
    /// The `Tensor` form of `freq_factors` is also retained on the
    /// struct for the fused kernel path (bound as `src2`).
    fn new(
        head_dim: usize,
        max_seq_len: usize,
        rope_theta: f64,
        freq_factors_host: Option<Vec<f32>>,
        dev: &Device,
        kernel: RopeKernel,
        counters: Arc<DispatchCounters>,
    ) -> Result<Self> {
        Self::build(
            head_dim,
            max_seq_len,
            rope_theta,
            freq_factors_host,
            dev,
            kernel,
            counters,
        )
    }

    fn build(
        head_dim: usize,
        max_seq_len: usize,
        rope_theta: f64,
        freq_factors_host: Option<Vec<f32>>,
        dev: &Device,
        kernel: RopeKernel,
        counters: Arc<DispatchCounters>,
    ) -> Result<Self> {
        let half = head_dim / 2;

        // Validate shape contract up-front so a bad GGUF never
        // silently slices past the end of the freq_factors vector.
        if let Some(ref ff) = freq_factors_host {
            if ff.len() != half {
                return Err(anyhow::anyhow!(
                    "rope_freqs.weight length {} does not match head_dim/2 = {}",
                    ff.len(),
                    half,
                ));
            }
        }

        // Loop-path sin/cos cache.
        //
        // llama.cpp's kernel computes `theta = pos * pow(freq_base,
        // -2*ic/head_dim) / freq_factor[ic]` for pair `ic ∈ [0, half)`.
        // hf2q's loop path pre-computes a `[max_seq_len, half]` table
        // from `freqs[pos, i] = pos * inv_freq[i]`, so we fold the
        // freq_factors division directly into `inv_freq[i]`. For
        // `freq_factors[i] = 1e+30`, `inv_freq_eff[i] ≈ 0`, which gives
        // `cos = 1, sin = 0` — identity rotation, matching the mask's
        // intent. Sliding layers pass `None` (treated as identity
        // division) so their cached values are byte-identical to the
        // pre-1bNEW.18 table.
        let inv_freq: Vec<f32> = (0..half)
            .map(|i| {
                let base = 1f32 / rope_theta.powf(2.0 * i as f64 / head_dim as f64) as f32;
                match freq_factors_host.as_ref() {
                    Some(ff) => base / ff[i],
                    None => base,
                }
            })
            .collect();
        let inv_freq = Tensor::from_vec(inv_freq, (1, half), dev)?;
        let t = Tensor::arange(0u32, max_seq_len as u32, dev)?
            .to_dtype(DType::F32)?
            .reshape((max_seq_len, 1))?;
        let freqs = t.matmul(&inv_freq)?;
        let sin = freqs.sin()?;
        let cos = freqs.cos()?;

        // Fused-path freq_factors buffer. Kept as an on-device F32
        // `[half]` tensor so the encoder can bind its Metal buffer
        // directly as `src2` without a per-dispatch host→device copy.
        let freq_factors_tensor = match freq_factors_host {
            Some(ff) => Some(Tensor::from_vec(ff, (half,), dev)?),
            None => None,
        };

        Ok(Self {
            sin,
            cos,
            head_dim,
            rope_theta: rope_theta as f32,
            freq_factors: freq_factors_tensor,
            // Gemma 4 is always Neox. If a future non-Gemma4 caller
            // needs Norm, add a `new_norm_variant` constructor rather
            // than threading it through the signature (Anti-Goal #7 —
            // no speculative flexibility).
            variant: RopeVariant::Neox,
            kernel,
            counters,
        })
    }

    fn apply(&self, q: &Tensor, k: &Tensor, seqlen_offset: usize) -> Result<(Tensor, Tensor)> {
        if self.kernel.is_fused() {
            // ADR-005 1bNEW.6 — single Metal dispatch per Q and per K
            // per layer. The ported `kernel_rope_neox` reads source
            // strides from the transposed `[B,H,S,D]` view directly
            // via `args.nb00..nb03`, so NO `.contiguous()` bounce is
            // required on either Q or K even though `transpose(1,2)`
            // leaves them non-contiguous. This absorbs the old
            // 1bNEW.8 stride-aware-prelude win (ADR-005:322-326).
            //
            // **ADR-005 1bNEW.18 (2026-04-11):** for global layers
            // `self.freq_factors = Some(rope_freqs_device_tensor)` is
            // propagated into the kernel via `src2`. The kernel branch
            // at `rope_kernel.rs:319` is already a byte-port of
            // llama.cpp's `freq_factor = (args.src2 != 0) ? src2[ic]
            // : 1.0f` gate — it was correct pre-landing; only the
            // caller needed fixing. Sliding layers pass
            // `freq_factors = None`, preserving identity-division
            // semantics exactly.
            //
            // Full-head rotation: both sliding and global pass
            // `n_dims = head_dim`, so llama.cpp's kernel rotates every
            // pair `(q[ic], q[ic + head_dim/2])` for ic ∈ [0, half).
            // No pass-through tail. The global mask's `1e+30` entries
            // at indices [64..256) collapse those pair rotations to
            // identity via the runtime `theta / freq_factor` division
            // inside the kernel, exactly matching the per-pair mask
            // behavior Gemma 4's GGUF encodes.
            let pipelines = self.kernel.pipelines.as_ref().expect("is_fused");
            let q_rot = rope_kernel::rope_fused(
                pipelines,
                q,
                seqlen_offset,
                self.head_dim,
                self.rope_theta,
                self.variant,
                self.freq_factors.as_ref(),
            )?;
            let k_rot = rope_kernel::rope_fused(
                pipelines,
                k,
                seqlen_offset,
                self.head_dim,
                self.rope_theta,
                self.variant,
                self.freq_factors.as_ref(),
            )?;
            // Two dispatches total for one Q + one K. The old loop
            // path used ~10 ops (see `fetch_add(10)` at the call site
            // below) — the per-token saving per layer is the old
            // outside-ops count (~10) minus 2 = ~8 per layer × 30 =
            // ~240 dispatches/token.
            self.counters.dispatches_per_token.fetch_add(2, Ordering::Relaxed);
            return Ok((q_rot, k_rot));
        }

        // Loop-mode dispatch accounting: the call site at
        // `Attention::forward` used to add `+10` as a conservative
        // upper bound; we moved the counting into this function so
        // the fused branch can add its own `+2` and the totals stay
        // consistent without the outside-site having to know which
        // path ran. Preserve the `10` ops accounting for loop mode
        // so the baseline metrics.txt numbers don't drift.
        //
        // **1bNEW.18:** the old loop path had two sub-branches
        // (`rotary_dim == head_dim` and `rotary_dim < head_dim`),
        // each counted as `10`. Post-landing there is only the
        // full-head branch, so the count stays `10`.
        self.counters.dispatches_per_token.fetch_add(10, Ordering::Relaxed);

        let (_b, _h, seq_len, head_dim) = q.dims4()?;
        debug_assert_eq!(head_dim, self.head_dim, "RoPE head_dim mismatch");
        let cos = self.cos.narrow(0, seqlen_offset, seq_len)?.to_dtype(q.dtype())?;
        let sin = self.sin.narrow(0, seqlen_offset, seq_len)?.to_dtype(q.dtype())?;

        // Full-head rotation — sin/cos tables were built with
        // `freq_factors` already folded into `inv_freq`, so for
        // global layers the pairs at ic ∈ [64..256) have
        // `sin ≈ 0, cos ≈ 1` and rotate to identity.
        let q_rot = Self::rope_apply(&q.contiguous()?, &cos, &sin)?;
        let k_rot = Self::rope_apply(&k.contiguous()?, &cos, &sin)?;
        Ok((q_rot, k_rot))
    }

    /// Standard RoPE rotation: split x into (x1,x2), rotate.
    fn rope_apply(x: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
        let half = x.dim(D::Minus1)? / 2;
        let x1 = x.narrow(D::Minus1, 0, half)?;
        let x2 = x.narrow(D::Minus1, half, half)?;
        let r1 = (x1.broadcast_mul(&cos)? - x2.broadcast_mul(&sin)?)?;
        let r2 = (x1.broadcast_mul(&sin)? + x2.broadcast_mul(&cos)?)?;
        Tensor::cat(&[r1, r2], D::Minus1).map_err(Into::into)
    }
}

// ---------------------------------------------------------------------------
// KV Cache (simple concat)
// ---------------------------------------------------------------------------

struct KvCache {
    k: Tensor,
    v: Tensor,
    cache_size: usize,
    current_len: usize,
    num_kv_heads: usize,
    head_dim: usize,
    sliding_window: Option<usize>,
    /// ADR-005 1bNEW.20 — dispatch mode for the append primitive.
    /// `SliceScatter` preserves the pre-1bNEW.20 path; `InPlace` uses
    /// `Tensor::slice_set`, mirroring llama.cpp's `llama_kv_cache::cpy_k`
    /// / `cpy_v` at `/opt/llama.cpp/src/llama-kv-cache.cpp:1196-1285`.
    mode: KvCacheKernelMode,
}

const KV_CACHE_INITIAL_SIZE: usize = 4096;

impl KvCache {
    fn new(
        num_kv_heads: usize,
        head_dim: usize,
        device: &Device,
        sliding_window: Option<usize>,
        mode: KvCacheKernelMode,
    ) -> Result<Self> {
        let cache_size = match sliding_window {
            Some(w) => KV_CACHE_INITIAL_SIZE.min(w),
            None => KV_CACHE_INITIAL_SIZE,
        };
        let k = Tensor::zeros((1, num_kv_heads, cache_size, head_dim), MODEL_DTYPE, device)?;
        let v = Tensor::zeros((1, num_kv_heads, cache_size, head_dim), MODEL_DTYPE, device)?;
        Ok(Self {
            k,
            v,
            cache_size,
            current_len: 0,
            num_kv_heads,
            head_dim,
            sliding_window,
            mode,
        })
    }

    /// Grow the pre-allocated cache if the next append would overflow.
    /// Shared between `append_slice_scatter` and `append_in_place`.
    ///
    /// Growth is rare (O(log n) reallocations over the full sequence);
    /// its cost does not affect the per-decode-step path either mode
    /// cares about. **Critical invariant:** after this function returns,
    /// `self.k` and `self.v` MUST be contiguous. `slice_set` (the
    /// in-place path's primitive) bails with "slice-set only supports
    /// contiguous tensors" if called against a non-contiguous `self`,
    /// which is what happens if we use `slice_scatter` on dim=2 to
    /// carry the active region into the new buffer — the dim-!=-0
    /// `slice_scatter` transpose-trick at `candle-core/src/tensor.rs:1723-1733`
    /// leaves a non-contiguous layout. We instead copy the active
    /// region via `slice_set` directly (an in-place `copy2d` into the
    /// zero-allocated new buffer), which preserves the contiguous
    /// layout of `new_k` / `new_v`.
    ///
    /// This invariant-preserving growth is load-bearing for the
    /// `InPlace` path; the `SliceScatter` path used to tolerate it
    /// because its post-append `.contiguous()` re-materialized a
    /// packed layout on every decode step, hiding the problem. The
    /// a0952e2 stride gotcha (ADR-005 line 229, bisect row 0) is
    /// precisely this class of bug.
    fn grow_if_needed(&mut self, needed: usize) -> Result<()> {
        if needed <= self.cache_size {
            return Ok(());
        }
        let mut new_size = self.cache_size;
        while new_size < needed {
            new_size *= 2;
        }
        let device = self.k.device().clone();
        let new_k = Tensor::zeros(
            (1, self.num_kv_heads, new_size, self.head_dim),
            MODEL_DTYPE,
            &device,
        )?;
        let new_v = Tensor::zeros(
            (1, self.num_kv_heads, new_size, self.head_dim),
            MODEL_DTYPE,
            &device,
        )?;
        if self.current_len > 0 {
            // Copy the active region into the new buffer. `slice_set`
            // requires a contiguous `src`; the existing cache IS
            // contiguous by this function's post-condition, so narrowing
            // on dim 2 yields a stride-!=-contiguous view, which means
            // we must `.contiguous()` it before the set. Growth is
            // already on the slow path (rare), so this extra copy is
            // acceptable and preserves the `new_k` / `new_v` layout.
            let active_k = self.k.narrow(2, 0, self.current_len)?.contiguous()?;
            let active_v = self.v.narrow(2, 0, self.current_len)?.contiguous()?;
            new_k.slice_set(&active_k, 2, 0)?;
            new_v.slice_set(&active_v, 2, 0)?;
        }
        self.k = new_k;
        self.v = new_v;
        self.cache_size = new_size;
        Ok(())
    }

    /// Dispatch entry point. Selects between the `SliceScatter` baseline
    /// and the `InPlace` 1bNEW.20 port based on `self.mode`.
    fn append(&mut self, k_new: &Tensor, v_new: &Tensor) -> Result<(Tensor, Tensor)> {
        match self.mode {
            KvCacheKernelMode::SliceScatter => self.append_slice_scatter(k_new, v_new),
            KvCacheKernelMode::InPlace => self.append_in_place(k_new, v_new),
        }
    }

    /// Phase-1 baseline — two `Tensor::slice_scatter` writes into the
    /// pre-allocated cache followed by `narrow` + `contiguous` on the
    /// active region (6 candle ops per layer per token on the hot path).
    ///
    /// The `.contiguous()` at the end is load-bearing: `slice_scatter`
    /// on dim != 0 uses a `transpose(0, dim).slice_scatter0().transpose(0, dim)`
    /// trick internally (see `candle-core/src/tensor.rs:1723-1733`) that
    /// leaves the returned tensor with non-standard strides — the memory
    /// layout is `[seq, heads, 1, hd]` but the shape is `[1, heads, seq, hd]`,
    /// so the position stride is `heads*hd` instead of `hd`. SDPA's
    /// vector kernel assumes positions are contiguous (stride = hd), so
    /// the caller MUST hand it a contiguous view. Removing the
    /// `.contiguous()` here is what caused the a0952e2 gibberish
    /// regression — see ADR-005 line 229 and the bisect table.
    fn append_slice_scatter(
        &mut self,
        k_new: &Tensor,
        v_new: &Tensor,
    ) -> Result<(Tensor, Tensor)> {
        let new_len = k_new.dim(2)?;
        let needed = self.current_len + new_len;
        self.grow_if_needed(needed)?;

        self.k = self.k.slice_scatter(k_new, 2, self.current_len)?;
        self.v = self.v.slice_scatter(v_new, 2, self.current_len)?;
        self.current_len = needed;

        let (visible_start, visible_len) = self.visible_range();

        // See method doc for why `.contiguous()` is load-bearing here.
        let k_active = self.k.narrow(2, visible_start, visible_len)?.contiguous()?;
        let v_active = self.v.narrow(2, visible_start, visible_len)?.contiguous()?;
        Ok((k_active, v_active))
    }

    /// ADR-005 1bNEW.20 — in-place append via `Tensor::slice_set`.
    /// Walk-KERNEL-PORT of llama.cpp's `llama_kv_cache::cpy_k` / `cpy_v`
    /// at `/opt/llama.cpp/src/llama-kv-cache.cpp:1196-1285`, which uses
    /// `ggml_set_rows` to write the new K/V slots directly into a view
    /// of the pre-allocated cache at a computed offset.
    ///
    /// **Why no `.contiguous()` on the returned view:** `slice_set`
    /// performs a direct `storage.copy2d` from `src` into `self`'s
    /// buffer at `offset * block_size` elements, preserving the
    /// pre-allocated cache's row-major layout untouched
    /// (`candle-core/src/tensor_cat.rs:286-299`). The returned active
    /// region is a plain `narrow` on dim 2 — its strides are the
    /// pre-allocated cache's strides (`[H*cache_size*hd, cache_size*hd, hd, 1]`
    /// in elements), which SDPA's vector kernel consumes directly via
    /// `k_stride[1]` and `v_stride[1]` (see
    /// `candle-metal-kernels/src/kernels/sdpa.rs:278-279`). The kernel
    /// walks only the first `visible_len` positions per head, so the
    /// non-`visible_len*hd` per-head stride is read correctly.
    ///
    /// **Prefill consumer contract:** prefill paths (`q_len > 1`) that
    /// must consume a contiguous layout — global bd=512 SDPA full path's
    /// `.to_dtype(BF16)?.contiguous()?` and sliding manual path's
    /// `reshape` after `unsqueeze/expand` — add their own `.contiguous()`
    /// on the view. That was already paid under the `SliceScatter` path
    /// (globally, via the cast-and-contiguous chain); on the sliding
    /// manual path the landing commit adds an explicit `.contiguous()`
    /// call before the reshape. Either way, the cost is paid only once
    /// per prefill pass, never on a decode step.
    ///
    /// **Input contiguity requirement:** `Tensor::slice_set` requires
    /// `src` to be contiguous (`tensor_cat.rs:248`). `k_new` arrives
    /// fresh from `rope_fused` (which allocates a contiguous output
    /// buffer at `rope_kernel.rs:704-712`), so it's contiguous with
    /// probability 1 on the fused path; the loop path's
    /// `rope_apply` output is also contiguous. `v_new` arrives from a
    /// `transpose(1, 2)` after `rms_norm_unit`, which leaves it
    /// non-contiguous — we force a contiguous copy before the write.
    /// This is 1 op; the savings come from eliminating the 2
    /// contiguous copies of the *entire* active region that the
    /// `SliceScatter` path pays at read time.
    fn append_in_place(&mut self, k_new: &Tensor, v_new: &Tensor) -> Result<(Tensor, Tensor)> {
        let new_len = k_new.dim(2)?;
        let needed = self.current_len + new_len;
        self.grow_if_needed(needed)?;

        // Both `slice_set` and `copy2d` require contiguous src. `k_new`
        // is always contiguous (fresh `rope_fused` output). `v_new` is
        // a transposed view and must be made contiguous before the
        // write — this is the only per-token copy the in-place path
        // pays, and it's a tight `[1, kv_heads, q_len, hd]` buffer
        // (tiny at decode, q_len=1).
        let k_src = if k_new.is_contiguous() {
            k_new.clone()
        } else {
            k_new.contiguous()?
        };
        let v_src = if v_new.is_contiguous() {
            v_new.clone()
        } else {
            v_new.contiguous()?
        };

        // In-place writes. `slice_set` returns `()` — it mutates the
        // underlying Metal buffer directly, preserving the cache's
        // `[1, kv_heads, cache_size, hd]` contiguous layout.
        self.k.slice_set(&k_src, 2, self.current_len)?;
        self.v.slice_set(&v_src, 2, self.current_len)?;
        self.current_len = needed;

        let (visible_start, visible_len) = self.visible_range();

        // Stride-aware view — NOT contiguous when `visible_len < cache_size`,
        // but the SDPA vector kernel reads strides explicitly and the
        // prefill paths handle contiguity at their own call site. See
        // the method doc above for the full analysis.
        let k_active = self.k.narrow(2, visible_start, visible_len)?;
        let v_active = self.v.narrow(2, visible_start, visible_len)?;
        Ok((k_active, v_active))
    }

    /// Sliding-window visibility range. Global layers (sliding_window=None)
    /// see the full history; sliding layers see only the last W tokens.
    /// Shared between both append paths.
    fn visible_range(&self) -> (usize, usize) {
        let visible_start = match self.sliding_window {
            Some(w) if self.current_len > w => self.current_len - w,
            _ => 0,
        };
        let visible_len = self.current_len - visible_start;
        (visible_start, visible_len)
    }

    #[allow(dead_code)]
    fn reset(&mut self) {
        self.current_len = 0;
    }
}

// ---------------------------------------------------------------------------
// QLinear layer (quantized weights via candle QMatMul)
// ---------------------------------------------------------------------------

struct QLinear {
    inner: QMatMul,
    bias: Option<Tensor>,
    counters: Arc<DispatchCounters>,
}

impl QLinear {
    fn new(qmatmul: QMatMul, bias: Option<Tensor>, counters: Arc<DispatchCounters>) -> Self {
        Self { inner: qmatmul, bias, counters }
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // Metal QMatMul kernels require F32 input; cast and cast back
        let in_dtype = x.dtype();
        let x_f32 = if in_dtype != DType::F32 { x.to_dtype(DType::F32)? } else { x.clone() };
        let out = self.inner.forward(&x_f32)?;
        let out = if in_dtype != DType::F32 { out.to_dtype(in_dtype)? } else { out };
        // At F32 steady state (MODEL_DTYPE == F32), the dtype casts short-circuit
        // to clones; the only real dispatches are the QMatMul itself and the
        // optional bias add.
        let mut ops: u64 = 1;
        let result = match &self.bias {
            Some(b) => { ops += 1; out.broadcast_add(b)? }
            None => out,
        };
        self.counters.dispatches_per_token.fetch_add(ops, Ordering::Relaxed);
        Ok(result)
    }
}

// ---------------------------------------------------------------------------
// Attention
// ---------------------------------------------------------------------------

struct Attention {
    q_proj: QLinear,
    k_proj: QLinear,
    v_proj: QLinear,
    o_proj: QLinear,
    q_norm: RmsNorm,
    k_norm: RmsNorm,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    rotary_emb: Arc<RotaryEmbedding>,
    kv_cache: KvCache,
    k_eq_v: bool,
    counters: Arc<DispatchCounters>,
    /// ADR-005 1bNEW.4 — passed to `rms_norm_unit` for the V-norm call.
    rms_kernel: RmsNormKernel,
}

impl Attention {
    fn forward(
        &mut self,
        xs: &Tensor,
        seqlen_offset: usize,
    ) -> Result<Tensor> {
        let (b_sz, q_len, _) = xs.dims3()?;

        let q_raw = self.q_proj.forward(xs)?;
        let q = q_raw.reshape((b_sz, q_len, self.num_heads, self.head_dim))?;
        let k_raw = self.k_proj.forward(xs)?;
        let k = k_raw.reshape((b_sz, q_len, self.num_kv_heads, self.head_dim))?;
        let v = if self.k_eq_v {
            k.clone()
        } else {
            self.v_proj.forward(xs)?
                .reshape((b_sz, q_len, self.num_kv_heads, self.head_dim))?
        };
        // 2 reshapes + 1 optional reshape (if not k_eq_v) — just metadata ops in
        // candle, but count as dispatches for discipline.
        self.counters.dispatches_per_token.fetch_add(
            if self.k_eq_v { 2 } else { 3 },
            Ordering::Relaxed,
        );

        // Apply Q/K norms before transpose
        let q = self.q_norm.forward(&q)?.transpose(1, 2)?;
        let k = self.k_norm.forward(&k)?.transpose(1, 2)?;
        // V gets a unit RMSNorm (just normalize, no learned weight)
        let v = rms_norm_unit(&v, &self.counters, &self.rms_kernel)?.transpose(1, 2)?;
        self.counters.dispatches_per_token.fetch_add(3, Ordering::Relaxed);

        // RoPE: `rotary_emb.apply` self-counts `+10` in loop mode
        // (matching the pre-1bNEW.6 fetch_add here) and `+2` in
        // fused mode. The counting moved INTO the function in
        // 1bNEW.6 so the Attention call site is mode-agnostic.
        let (q, k) = self.rotary_emb.apply(&q, &k, seqlen_offset)?;

        // KV cache append — ADR-005 1bNEW.20 mode-aware dispatch count.
        // `SliceScatter` counts 6 ops (2 `slice_scatter` + 2 `narrow` +
        // 2 `contiguous`). `InPlace` counts 3 ops (1 `v.contiguous()`
        // + 2 `slice_set`; the 2 `narrow` returns are pure view ops
        // with zero Metal dispatches). The reduction from 6→3 per
        // layer per token aligns with the measured +26.7 tok/s speed
        // win; see ADR-005 1bNEW.20 Phase B bench table.
        let kv_ops: u64 = match self.kv_cache.mode {
            KvCacheKernelMode::SliceScatter => 6,
            KvCacheKernelMode::InPlace => 3,
        };
        let (k, v) = self.kv_cache.append(&k, &v)?;
        self.counters
            .dispatches_per_token
            .fetch_add(kv_ops, Ordering::Relaxed);

        // ADR-005 1bNEW.10 — BF16 prefill SDPA at head_dim=512 (Walk item, 2026-04-10,
        // refined post-9cc522d empirical measurement to split by head_dim).
        //
        // Decode (`q_len == 1`) routes to candle's SDPA vector path in F32 —
        // unchanged from Phase 1 1b.8; GQA is native; no `repeat_kv` copy.
        //
        // Prefill (`q_len > 1`) is split by head_dim:
        //   - **Global layers (head_dim=512, bd=512):** cast Q/K/V to BF16 and
        //     dispatch through candle's fused SDPA full kernel. Eliminates the
        //     10-12-dispatch manual `repeat_kv + matmul + softmax + matmul`
        //     chain. Q4 spike (`docs/spike-Q3Q4Q5-results.md`) pre-validated:
        //     argmax identical, top-5 order identical, max |Δp post-softmax|
        //     ≤ 1.12e-3, needle-recall preserved on a 638-token adversarial
        //     prompt.
        //   - **Sliding layers (head_dim=256, bd=256):** retain the pre-existing
        //     manual `repeat_kv + matmul + causal_mask + softmax + matmul`
        //     chain. Candle's bd=256 SDPA full kernel is **unusable** at this
        //     head_dim for BOTH dtypes:
        //       * bd=256 F32 exceeds the 32 KB threadgroup memory limit at
        //         53_760 B (runtime `AGXMetalG17X` crash). Compiled but
        //         unusable.
        //       * bd=256 BF16 produces NaN on q_seq values in the sawtooth
        //         bands [13..16, 33..48, ...] — a bug inside the fused
        //         attention kernel template in
        //         `candle-metal-kernels/src/metal_src/scaled_dot_product_attention.metal:1895-1970`.
        //         The 187-token canonical bench and 638-token Q4 spike prompt
        //         both happened to land in OK bands.
        //     Neither blocker is hf2q's code to fix; both are upstream candle
        //     gaps owned by a follow-up item (candle upstream fix OR port of
        //     llama.cpp's flash-attn vec kernel for head_dim=256).
        //
        // Why the split is still a net Walk win:
        //   1. The Walk Exception Register entry at ADR-005 line 141 is about
        //      the prefill mask shape mismatch at `q_len > sliding_window`.
        //      This item does not RESOLVE that exception — sliding layers
        //      retain the same square causal mask that mismatches at
        //      q_len>1024, same envelope as pre-1bNEW.10. It is NOT a
        //      regression.
        //   2. Global layers ARE now fused at prefill — fewer dispatches per
        //      global layer, ~5 MB less temporary `repeat_kv` allocation per
        //      prefill pass, and a smaller per-prefill critical path on the
        //      global attention QMatMul cliff the Q5 spike already measured.
        //   3. The BF16 numerical drift on bd=512 is bounded by the Q4 spike
        //      at ε ≤ 1e-3 on the needle prompt and does not flip the top-1
        //      on the canonical bench prompt.
        //
        // Citations (all read end-to-end on disk):
        //   - `candle-metal-kernels/src/kernels/sdpa.rs:86-94` — bd=512
        //     reduced-tile selection and F32 rejection.
        //   - `candle-metal-kernels/src/metal_src/scaled_dot_product_attention.metal:2332-2337`
        //     — bd=512 compiled instantiations for f16/bf16 × matching-float/bool.
        //   - `candle-nn/src/ops.rs:1178-1179` — `mask_type == itype` constraint
        //     that `mask=None, do_causal=true` sidesteps.
        //   - `candle-nn/src/ops.rs:1261-1280` — public `sdpa` signature.
        //   - ADR-005 Q4 resolution (lines 723) — BF16 prefill spike PASS.
        //   - ADR-005 Q8 resolution (line 726) — mask rework with do_causal=true.
        let attn_out = if q_len == 1 {
            // One SDPA dispatch. Decode path unchanged: F32 in, F32 out.
            self.counters.dispatches_per_token.fetch_add(1, Ordering::Relaxed);
            candle_nn::ops::sdpa(&q, &k, &v, None, false, 1.0, 1.0)?
        } else if self.head_dim == 512 {
            // Global attention layers (bd=512): cast Q/K/V to BF16 and dispatch
            // through candle's fused SDPA full kernel. bd=512 is the **only**
            // prefill head_dim where candle's fused path is usable for hf2q:
            //   - `candle-metal-kernels/src/kernels/sdpa.rs:86-94` selects the
            //     reduced tile (bq=8, bk=8, wm=1, wn=1) for bd=512, totalling
            //     24.1 KB threadgroup memory (fits 32 KB limit).
            //   - F32 is rejected at `sdpa.rs:87-92` (the F32 full tile at
            //     bd=512 would exceed 32 KB), so BF16/F16 is mandatory.
            //   - The compiled `bfloat16` instantiation lives at
            //     `candle-metal-kernels/src/metal_src/scaled_dot_product_attention.metal:2336-2337`.
            //   - `mask=None, do_causal=true` sidesteps the
            //     `mask_type == itype` constraint at
            //     `candle-nn/src/ops.rs:1178-1179` AND the
            //     `q_seq <= k_seq` mask constraint at `ops.rs:1041`.
            //   - Q4 spike (`docs/spike-Q3Q4Q5-results.md`) pre-validated this
            //     path at 187 and 638 tokens: argmax identical, top-5 order
            //     identical, max |Δp post-softmax| ≤ 1.12e-3.
            let q_bf16 = q.to_dtype(DType::BF16)?.contiguous()?;
            let k_bf16 = k.to_dtype(DType::BF16)?.contiguous()?;
            let v_bf16 = v.to_dtype(DType::BF16)?.contiguous()?;
            let out_bf16 = candle_nn::ops::sdpa(&q_bf16, &k_bf16, &v_bf16, None, true, 1.0, 1.0)?;
            // 3 dtype casts + 3 contiguous + 1 SDPA + 1 cast back = 8 ops.
            self.counters.dispatches_per_token.fetch_add(8, Ordering::Relaxed);
            out_bf16.to_dtype(DType::F32)?
        } else {
            // Sliding attention layers (head_dim=256): **cannot** use candle's
            // SDPA full kernel at this head_dim during prefill. Two blockers
            // measured 2026-04-10 on top of commit `9cc522d` and confirmed
            // empirically:
            //
            //   1. **bd=256 F32 threadgroup memory blowup.** candle's bd=256
            //      tile uses (bq=32, bk=16, wm=4, wn=1). Q_smem alone is
            //      `BQ*(BD+padQ) = 32*(256+4)*sizeof(float) = 33280 B`; total
            //      with KV_smem is 53760 B. The kernel aborts at dispatch:
            //      `AGXMetalG17X "Threadgroup memory size (53760) exceeds the
            //      maximum threadgroup memory allowed (32768)"`. F32 is
            //      compile-time instantiated at
            //      `candle-metal-kernels/src/metal_src/scaled_dot_product_attention.metal:2330`
            //      but runtime-unusable at bd=256. The reduced-tile selection
            //      at `candle-metal-kernels/src/kernels/sdpa.rs:86-98` only
            //      guards bd=512; bd=256 F32 silently compiles and crashes on
            //      dispatch. Upstream candle gap.
            //   2. **bd=256 BF16 kernel produces NaN on a sawtooth of q_seq
            //      values.** Measured post-commit 9cc522d: q_seq ∈ [13..16,
            //      33..48] → NaN; q_seq ∈ [17..32, 49..] → OK. The 187-token
            //      canonical bench and 638-token Q4 spike prompt both land in
            //      the OK band, which is why the Q4 spike did not surface it.
            //      Root cause is inside candle's fused attention kernel
            //      template at `scaled_dot_product_attention.metal:1895-1970`,
            //      specifically in the final partial q-row batch. Upstream
            //      fix out of scope for 1bNEW.10.
            //
            // Consequence: sliding-layer prefill retains the pre-existing
            // manual `repeat_kv + matmul + causal_mask_add + softmax + matmul`
            // chain. We are strictly **adding** the bd=512 fused SDPA path
            // (global layers) without regressing bd=256 sliding-layer
            // correctness. Same F32 dispatches as before 1bNEW.10, same
            // shape-mismatch hazard at q_len > sliding_window, same
            // correctness envelope — no regression at any prompt length
            // ≤ sliding_window=1024.
            //
            // Walk exception: this divergence from the original 1bNEW.10 plan
            // ("cast Q/K/V to BF16 before calling candle_nn::ops::sdpa" — one
            // global path for both head_dims) is logged in the ADR-005 Walk
            // Exception Register by the ADR update commit. The bd=256 fused
            // prefill path is handed off to a follow-up item (either candle
            // upstream fix, or 1bNEW.11-style port of llama.cpp's vec kernel
            // for head_dim=256).
            let gqa = self.num_heads / self.num_kv_heads;
            let k_exp = if gqa == 1 {
                k.clone()
            } else {
                let (b, kh, s, d) = k.dims4()?;
                k.unsqueeze(2)?
                    .expand((b, kh, gqa, s, d))?
                    .reshape((b, kh * gqa, s, d))?
            };
            let v_exp = if gqa == 1 {
                v.clone()
            } else {
                let (b, kh, s, d) = v.dims4()?;
                v.unsqueeze(2)?
                    .expand((b, kh, gqa, s, d))?
                    .reshape((b, kh * gqa, s, d))?
            };
            let attn_weights = q.matmul(&k_exp.transpose(D::Minus2, D::Minus1)?)?;
            // Build a square causal mask matching pre-1bNEW.10 semantics. At
            // `q_len ≤ sliding_window = 1024` the sliding KV cache exposes
            // exactly `q_len` positions after prefill (visible_start = 0 in
            // `KvCache::append`), so kv_seq == q_seq and the square lower-
            // triangular mask is semantically correct — identical to the
            // pre-1bNEW.10 causal_mask at `seqlen_offset=0`. At `q_len >
            // sliding_window` the KV is truncated to the last 1024 positions,
            // causing a `broadcast_add` shape mismatch — the same hazard
            // documented in the ADR-005 Walk Exception Register at line 141.
            // This item does not claim to fix the sliding-window semantic
            // bug; it claims to fix the head_dim=512 prefill path fidelity
            // and leave head_dim=256 at the same envelope as pre-1bNEW.10.
            let (_, _, q_seq, _) = q.dims4()?;
            let dev = q.device();
            let mask_vec: Vec<f32> = (0..q_seq)
                .flat_map(|i| {
                    (0..q_seq).map(move |j| {
                        if j > i { f32::NEG_INFINITY } else { 0.0 }
                    })
                })
                .collect();
            let m = Tensor::from_vec(mask_vec, (1, 1, q_seq, q_seq), dev)?;
            let attn_weights = attn_weights.broadcast_add(&m)?;
            let attn_weights = candle_nn::ops::softmax_last_dim(&attn_weights)?;
            let out = attn_weights.matmul(&v_exp)?;
            // repeat_kv × 2 (3 ops each) + matmul + transpose + mask-build +
            // broadcast_add + softmax + matmul ≈ 12 ops.
            self.counters.dispatches_per_token.fetch_add(12, Ordering::Relaxed);
            out
        };

        // Reshape back (transpose + reshape = 2 ops)
        let attn_out = attn_out
            .transpose(1, 2)?
            .reshape((b_sz, q_len, self.num_heads * self.head_dim))?;
        self.counters.dispatches_per_token.fetch_add(2, Ordering::Relaxed);

        self.o_proj.forward(&attn_out)
    }

    #[allow(dead_code)]
    fn clear_cache(&mut self) {
        self.kv_cache.reset();
    }
}

/// Unit RMSNorm (no learned weight, just normalize). Extracted out of
/// `Attention` as a free function so both `Attention::forward` and
/// `MoeBlock::forward` can share it while still counting dispatches into the
/// caller's counter Arc.
///
/// ADR-005 1bNEW.4: accepts a `&RmsNormKernel` so the caller can
/// dispatch on `kernel.mode` without a global switch. `Fused` routes to
/// `rms_norm_kernel::rms_norm_fused(.., weight=None, residual=None, eps=1e-6)`
/// (F=1); `Loop` runs the pre-1bNEW.4 9-op manual chain unchanged.
fn rms_norm_unit(
    x: &Tensor,
    counters: &Arc<DispatchCounters>,
    kernel: &RmsNormKernel,
) -> Result<Tensor> {
    if kernel.is_fused() {
        let pipelines = kernel.pipelines.as_ref().expect("is_fused");
        let x_c = if x.is_contiguous() { x.clone() } else { x.contiguous()? };
        let out = rms_norm_kernel::rms_norm_fused(
            pipelines,
            &x_c,
            None,
            None,
            1e-6_f32,
        )?;
        counters.norm_dispatches_per_token.fetch_add(1, Ordering::Relaxed);
        counters.dispatches_per_token.fetch_add(1, Ordering::Relaxed);
        return Ok(out);
    }

    let dtype = x.dtype();
    let x_f32 = x.to_dtype(DType::F32)?;
    let sq = x_f32.sqr()?;
    let mean_sq = sq.mean_keepdim(D::Minus1)?;
    let eps_t = mean_sq.ones_like()?.affine(0.0, 1e-6)?;
    let rms = (mean_sq + eps_t)?.sqrt()?.recip()?;
    let normed = x_f32.broadcast_mul(&rms)?;
    let out = normed.to_dtype(dtype)?;
    // 9 candle ops (no weight broadcast_mul vs RmsNorm::forward).
    counters.norm_dispatches_per_token.fetch_add(9, Ordering::Relaxed);
    counters.dispatches_per_token.fetch_add(9, Ordering::Relaxed);
    Ok(out)
}

// ---------------------------------------------------------------------------
// Dense MLP (SwiGLU)
// ---------------------------------------------------------------------------

struct Mlp {
    gate_proj: QLinear,
    up_proj: QLinear,
    down_proj: QLinear,
    counters: Arc<DispatchCounters>,
}

impl Mlp {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let gate = self.gate_proj.forward(x)?;
        let gate = candle_nn::Activation::GeluPytorchTanh.forward(&gate)?;
        let up = self.up_proj.forward(x)?;
        let fused = (gate * up)?;
        let out = self.down_proj.forward(&fused)?;
        // gelu (1) + elementwise mul (1) — the QMatMul calls self-count via
        // QLinear::forward.
        self.counters.dispatches_per_token.fetch_add(2, Ordering::Relaxed);
        Ok(out)
    }
}

// ---------------------------------------------------------------------------
// MoE (Mixture of Experts) with Softmax Routing
// ---------------------------------------------------------------------------

struct MoeBlock {
    /// Router projection: hidden_size → num_experts
    router_proj: QLinear,
    /// Per-hidden-dim learned scale for router input
    router_scale: Tensor,
    /// Per-expert scale applied to selected weights after softmax.
    /// Used by the `Fused` path (GPU gather by top_k_indices); the `Loop`
    /// path reads `per_expert_scale_cpu` instead.
    per_expert_scale: Tensor,
    /// Cached CPU copy of per_expert_scale (avoids GPU→CPU sync every forward).
    /// Used ONLY by the `Loop` path. The `Fused` path gathers from the
    /// GPU `per_expert_scale` tensor above.
    per_expert_scale_cpu: Vec<f32>,
    /// Per-expert gate_up QMatMul (Loop path fallback — byte-sliced from the
    /// 3D GGUF source tensor).
    expert_gate_up: Vec<QMatMul>,
    expert_down: Vec<QMatMul>,
    num_experts: usize,
    top_k: usize,
    moe_intermediate_size: usize,
    hidden_size: usize,
    counters: Arc<DispatchCounters>,

    // --- ADR-005 1bNEW.1 fused dispatch state (populated iff use_fused) ---
    /// Raw 3D gate_up weight storage `[num_experts, 2*moe_intermediate, hidden]`
    /// retained alongside the per-expert `Vec<QMatMul>` so the fused path
    /// can pass its Metal buffer pointer directly to `kernel_mul_mv_id_*`.
    /// `None` when this layer runs in Loop mode.
    #[cfg(feature = "metal")]
    fused_gate_up: Option<candle_core::quantized::QStorage>,
    /// Raw 3D down weight storage `[num_experts, hidden, moe_intermediate]`.
    #[cfg(feature = "metal")]
    fused_down: Option<candle_core::quantized::QStorage>,
    /// GGML dtype of the gate_up 3D storage — selects the fused kernel
    /// symbol (`kernel_mul_mv_id_q6_K_f32` etc.). Only meaningful if
    /// `fused_gate_up` is `Some`.
    #[cfg(feature = "metal")]
    fused_gate_up_dtype: MetalGgmlDType,
    #[cfg(feature = "metal")]
    fused_down_dtype: MetalGgmlDType,
    /// Which dispatch path this layer runs. `Loop` = Phase-1 baseline,
    /// `Fused` = ADR-005 1bNEW.1 fused kernel.
    mode: MoeKernelMode,
    /// ADR-005 1bNEW.4 — passed to `rms_norm_unit` for the router norm.
    rms_kernel: RmsNormKernel,
}

impl MoeBlock {
    /// Forward pass. `x` is the pre-normed expert input, `router_input` is
    /// the raw residual for the router (router applies its own RMS norm).
    ///
    /// Dispatches on `self.mode`:
    ///   - `Loop`:  Phase-1 baseline, per-expert `QMatMul::forward` loop.
    ///   - `Fused`: ADR-005 1bNEW.1 fused `kernel_mul_mv_id_*` dispatch.
    fn forward(&self, x: &Tensor, router_input: &Tensor) -> Result<Tensor> {
        // Track per-layer invocation; moe_dispatches is a total across all
        // MoeBlock forward calls this pass, and `per_token()` divides by
        // moe_layer_invocations to report per-layer.
        self.counters.moe_layer_invocations.fetch_add(1, Ordering::Relaxed);
        let start_dispatches = self.counters.dispatches_per_token.load(Ordering::Relaxed);

        // Router + GPU top-k. Both paths share this prefix — it stays on
        // the GPU end-to-end (no `to_vec2`). The `Loop` path then drains
        // the top-k indices/weights to CPU for its sequential expert loop;
        // the `Fused` path keeps them on the GPU and passes the index
        // buffer straight into the kernel.
        let (b_sz, seq_len, hidden) = x.dims3()?;
        let x_flat = x.reshape((b_sz * seq_len, hidden))?;
        let router_flat = router_input.reshape((b_sz * seq_len, hidden))?;
        self.counters.dispatches_per_token.fetch_add(2, Ordering::Relaxed);

        let router_normed = rms_norm_unit(&router_flat, &self.counters, &self.rms_kernel)?;
        let scale_factor = (self.hidden_size as f64).powf(-0.5);
        let router_scaled = (router_normed.broadcast_mul(&self.router_scale)? * scale_factor)?;
        self.counters.dispatches_per_token.fetch_add(2, Ordering::Relaxed);
        let logits = self.router_proj.forward(&router_scaled)?;

        let probs = candle_nn::ops::softmax_last_dim(&logits)?;
        self.counters.dispatches_per_token.fetch_add(1, Ordering::Relaxed);

        let probs_f32 = probs.to_dtype(DType::F32)?;
        let sorted_indices = probs_f32.contiguous()?.arg_sort_last_dim(false)?;
        let top_k_indices = sorted_indices.narrow(D::Minus1, 0, self.top_k)?.contiguous()?;
        let top_k_probs = probs_f32.contiguous()?.gather(&top_k_indices, D::Minus1)?;
        let top_k_sum = top_k_probs.sum_keepdim(D::Minus1)?;
        let top_k_weights = top_k_probs.broadcast_div(&top_k_sum)?;
        self.counters.dispatches_per_token.fetch_add(9, Ordering::Relaxed);

        let out = match self.mode {
            MoeKernelMode::Loop => {
                self.forward_loop(&x_flat, &top_k_indices, &top_k_weights, b_sz, seq_len, hidden, x.dtype())?
            }
            MoeKernelMode::Fused => {
                #[cfg(feature = "metal")]
                {
                    self.forward_fused(&x_flat, &top_k_indices, &top_k_weights, b_sz, seq_len, hidden, x.dtype())?
                }
                #[cfg(not(feature = "metal"))]
                {
                    // Walk discipline: no CPU/non-metal fallback
                    // (feedback_gpu_everything.md). If metal is disabled,
                    // `Fused` is an invalid configuration, not a silent
                    // downgrade.
                    anyhow::bail!(
                        "MoeKernelMode::Fused requires the `metal` feature. \
                         Rebuild with `--features metal` or pass `--moe-kernel=loop`."
                    );
                }
            }
        };

        // Attribute every dispatch counted during this MoeBlock::forward call
        // to `moe_dispatches` as well. We subtract the starting value so we
        // capture only the delta — this keeps `moe_dispatches` an exact
        // subset of `dispatches_per_token` restricted to MoE scope.
        let end_dispatches = self.counters.dispatches_per_token.load(Ordering::Relaxed);
        let moe_delta = end_dispatches.saturating_sub(start_dispatches);
        self.counters.moe_dispatches.fetch_add(moe_delta, Ordering::Relaxed);

        Ok(out)
    }

    /// Phase-1 baseline expert dispatch path. Drains `top_k_indices` /
    /// `top_k_weights` to CPU via two forced `to_vec2()` syncs (the very
    /// ones instrumented by `moe_to_vec2_count`), then runs a CPU-driven
    /// per-token, per-expert `QMatMul::forward` loop.
    ///
    /// Preserved verbatim from commit `6ff446e` so the `loop` mode stays
    /// byte-identical to the Phase-1 baseline — the `fused` mode is
    /// compared against THIS path in Phase A/B token-match gates.
    fn forward_loop(
        &self,
        x_flat: &Tensor,
        top_k_indices: &Tensor,
        top_k_weights: &Tensor,
        b_sz: usize,
        seq_len: usize,
        hidden: usize,
        out_dtype: DType,
    ) -> Result<Tensor> {
        let top_k_indices_cpu: Vec<Vec<u32>> = top_k_indices.to_vec2()?;
        let top_k_weights_cpu: Vec<Vec<f32>> = top_k_weights.to_vec2()?;
        self.counters.moe_to_vec2_count.fetch_add(2, Ordering::Relaxed);
        self.counters.dispatches_per_token.fetch_add(2, Ordering::Relaxed);

        let per_expert_scale_cpu = &self.per_expert_scale_cpu;
        let num_tokens = b_sz * seq_len;
        let device = x_flat.device();
        let mut outputs: Vec<Tensor> = Vec::with_capacity(num_tokens);

        for tok_idx in 0..num_tokens {
            let token_vec = x_flat.narrow(0, tok_idx, 1)?; // [1, hidden]
            let token_f32 = token_vec.to_dtype(DType::F32)?;

            let mut combined = Tensor::zeros((1, hidden), DType::F32, device)?;
            self.counters.dispatches_per_token.fetch_add(3, Ordering::Relaxed);
            for k in 0..self.top_k {
                let eid = top_k_indices_cpu[tok_idx][k] as usize;
                let w = top_k_weights_cpu[tok_idx][k] * per_expert_scale_cpu[eid];

                let gate_up_out = self.expert_gate_up[eid].forward(&token_f32)?;
                let gate = gate_up_out.narrow(1, 0, self.moe_intermediate_size)?;
                let up = gate_up_out.narrow(1, self.moe_intermediate_size, self.moe_intermediate_size)?;
                let gate_act = candle_nn::Activation::GeluPytorchTanh.forward(&gate)?;
                let fused = (gate_act * up)?;

                let expert_out = self.expert_down[eid].forward(&fused)?;

                let w_t = Tensor::new(&[w], device)?;
                combined = (combined + expert_out.broadcast_mul(&w_t)?)?;
                self.counters.dispatches_per_token.fetch_add(9, Ordering::Relaxed);
            }

            outputs.push(combined.to_dtype(out_dtype)?);
            self.counters.dispatches_per_token.fetch_add(1, Ordering::Relaxed);
        }

        let result = Tensor::cat(&outputs, 0)?;
        let out = result.reshape((b_sz, seq_len, hidden))?;
        self.counters.dispatches_per_token.fetch_add(2, Ordering::Relaxed);
        Ok(out)
    }

    /// ADR-005 1bNEW.1 fused expert dispatch.
    ///
    /// Replaces the CPU-driven per-expert loop with two fused
    /// `kernel_mul_mv_id_*` Metal dispatches (one for gate_up, one for
    /// down) plus candle Tensor ops for GELU, SwiGLU, scale gather and
    /// top-k reduction. The routing sync (`to_vec2`) is eliminated: the
    /// top-k index and weight tensors stay resident on the GPU and are
    /// consumed directly by the kernel / downstream broadcast.
    ///
    /// Shape summary (n_tokens = `b_sz * seq_len`):
    ///   x_flat:        [n_tokens, hidden]        (f32 or bf16)
    ///   top_k_indices: [n_tokens, top_k]         (u32 — byte-identical to i32)
    ///   top_k_weights: [n_tokens, top_k]         (f32)
    ///   gate_up_out:   [n_tokens, top_k, 2*int]  (f32, from fused kernel)
    ///   gate/up:       [n_tokens, top_k, int]    (slices of gate_up_out)
    ///   fused:         [n_tokens, top_k, int]    (GELU(gate) * up)
    ///   down_out:      [n_tokens, top_k, hidden] (f32, from fused kernel)
    ///   scale:         [n_tokens, top_k]         (GPU gather of per_expert_scale)
    ///   w_total:       [n_tokens, top_k, 1]      (top_k_weights * scale, broadcast-ready)
    ///   weighted:      [n_tokens, top_k, hidden] (down_out * w_total)
    ///   out:           [b_sz, seq_len, hidden]   (sum over top_k, reshape)
    ///
    /// Reference citations (verified in candle's vendored `quantized.metal`
    /// and llama.cpp host code):
    ///   - candle `kernel_mul_mv_id` template: quantized.metal:7544-7618
    ///   - llama.cpp host caller (kargs layout):
    ///     ggml-metal-ops.cpp:2393-2414
    ///   - mlx-lm dispatch shape cross-ref:
    ///     mlx_lm/models/switch_layers.py:76-90 (`mx.gather_qmm`)
    #[cfg(feature = "metal")]
    fn forward_fused(
        &self,
        x_flat: &Tensor,
        top_k_indices: &Tensor,
        top_k_weights: &Tensor,
        b_sz: usize,
        seq_len: usize,
        hidden: usize,
        out_dtype: DType,
    ) -> Result<Tensor> {
        use candle_core::quantized::QStorage;
        use candle_core::Storage;

        let num_tokens = b_sz * seq_len;
        let top_k = self.top_k;
        let intermediate = self.moe_intermediate_size;

        // --- Fused gate_up dispatch (Q6_K kernel_mul_mv_id) --------------
        // The input must be f32 and contiguous. Cast + contiguous lazily.
        let x_f32 = x_flat.to_dtype(DType::F32)?.contiguous()?;
        // `x_f32` is `[num_tokens, hidden]`.
        self.counters.dispatches_per_token.fetch_add(2, Ordering::Relaxed);

        // Pull the underlying Metal buffers from the candle tensors via the
        // public `storage_and_layout` hatch.
        let (x_storage, x_layout) = x_f32.storage_and_layout();
        let Storage::Metal(x_metal) = &*x_storage else {
            anyhow::bail!("forward_fused: x_f32 must be on Metal");
        };
        let src1_buf = x_metal.buffer().clone();
        let src1_offset = x_layout.start_offset() * DType::F32.size_in_bytes();

        let ids_contig = top_k_indices.contiguous()?;
        let (ids_storage, ids_layout) = ids_contig.storage_and_layout();
        let Storage::Metal(ids_metal) = &*ids_storage else {
            anyhow::bail!("forward_fused: top_k_indices must be on Metal");
        };
        let ids_buf = ids_metal.buffer().clone();
        // `top_k_indices` is U32; the kernel reads the same 32-bit slots
        // as i32, which is safe for positive expert ids < num_experts
        // (see Q1 resolution in ADR-005 lines 526-527).
        assert_eq!(ids_layout.start_offset(), 0,
            "forward_fused: top_k_indices must start at offset 0");

        // Metal device plumbing.
        let Device::Metal(metal_device) = x_f32.device() else {
            anyhow::bail!("forward_fused: device must be Metal");
        };
        let metal_device = metal_device.clone();

        // --- Fetch the 3D gate_up and down buffers -----------------------
        let gu_storage = self.fused_gate_up.as_ref()
            .expect("forward_fused: fused_gate_up must be Some when mode == Fused");
        let QStorage::Metal(gu_metal) = gu_storage else {
            anyhow::bail!("forward_fused: fused_gate_up must be a Metal QStorage");
        };
        let dn_storage = self.fused_down.as_ref()
            .expect("forward_fused: fused_down must be Some when mode == Fused");
        let QStorage::Metal(dn_metal) = dn_storage else {
            anyhow::bail!("forward_fused: fused_down must be a Metal QStorage");
        };

        // --- Allocate gate_up destination buffer -------------------------
        // Output shape [num_tokens, top_k, 2*intermediate], f32.
        let gu_out_elems = num_tokens * top_k * (2 * intermediate);
        let gu_dst_buf = metal_device.new_buffer(
            gu_out_elems, DType::F32, "moe_fused_gate_up_dst",
        )?;

        // --- Dispatch gate_up fused kernel -------------------------------
        {
            let encoder = metal_device.command_encoder()?;
            moe_kernel::call_quantized_matmul_mv_id_t(
                metal_device.device(),
                &encoder,
                metal_device.kernels(),
                self.fused_gate_up_dtype,
                moe_kernel::MoeDispatchShape {
                    num_experts: self.num_experts,
                    n: 2 * intermediate,
                    k: hidden,
                    n_tokens: num_tokens,
                    n_expert_used: top_k,
                },
                gu_metal.buffer(),
                &src1_buf,
                src1_offset,
                &ids_buf,
                &gu_dst_buf,
                0,
            ).map_err(|e| anyhow::anyhow!("fused gate_up dispatch: {e:?}"))?;
            // Compute encoder drops at block end — see Phase A notes.
        }
        self.counters.dispatches_per_token.fetch_add(1, Ordering::Relaxed);

        // Wrap the gate_up dst buffer in a candle Tensor so we can use the
        // regular Tensor op graph (narrow, gelu, mul, etc.) for the rest
        // of the fused path. This does NOT copy — the Tensor aliases the
        // buffer we just wrote.
        let gu_storage = candle_core::MetalStorage::new(
            gu_dst_buf,
            metal_device.clone(),
            gu_out_elems,
            DType::F32,
        );
        let gate_up_out = Tensor::from_storage(
            Storage::Metal(gu_storage),
            (num_tokens, top_k, 2 * intermediate),
            candle_core::op::BackpropOp::none(),
            false,
        );

        // --- SwiGLU: gate, up → GELU(gate) * up --------------------------
        let gate = gate_up_out.narrow(D::Minus1, 0, intermediate)?;
        let up = gate_up_out.narrow(D::Minus1, intermediate, intermediate)?;
        let gate_act = candle_nn::Activation::GeluPytorchTanh.forward(&gate)?;
        let swiglu = (gate_act * up)?.contiguous()?; // [num_tokens, top_k, intermediate]
        self.counters.dispatches_per_token.fetch_add(4, Ordering::Relaxed);

        // --- Fused down dispatch (Q8_0 kernel_mul_mv_id) -----------------
        // Input to the down kernel is `swiglu` flattened to 2D.
        let swiglu_2d = swiglu.reshape((num_tokens * top_k, intermediate))?;
        let (sw_storage, sw_layout) = swiglu_2d.storage_and_layout();
        let Storage::Metal(sw_metal) = &*sw_storage else {
            anyhow::bail!("forward_fused: swiglu must be on Metal");
        };
        let sw_buf = sw_metal.buffer().clone();
        let sw_offset = sw_layout.start_offset() * DType::F32.size_in_bytes();

        // IMPORTANT: the down kernel needs one input row per (token, slot)
        // pair, not one per token. To reuse the same `kernel_mul_mv_id`
        // wrapper, we bind `ids` as an identity index buffer
        // `[num_tokens*top_k, 1] = [0, 1, 2, ..., num_tokens*top_k - 1]`
        // with `n_expert_used = 1` and `n_tokens = num_tokens*top_k`. That
        // tells the kernel to index expert `i` using swiglu row `i` and
        // write to dst row `i`. Wait — that's wrong: we need each (token,
        // slot) to dispatch to a DIFFERENT expert, which is `top_k_indices`.
        //
        // Correct approach: use `top_k_indices` flat-shaped to
        // `[num_tokens*top_k, 1]` so `n_expert_used = 1` (one expert per
        // swiglu row) and `n_tokens = num_tokens*top_k`. The swiglu rows
        // are already arranged as [tok0_slot0, tok0_slot1, tok1_slot0, ...]
        // which is the same order the gate_up kernel wrote them in — and
        // the same order a flat view of `top_k_indices` iterates.
        let ids_flat = top_k_indices.reshape((num_tokens * top_k, 1))?.contiguous()?;
        let (idf_storage, idf_layout) = ids_flat.storage_and_layout();
        let Storage::Metal(idf_metal) = &*idf_storage else {
            anyhow::bail!("forward_fused: ids_flat must be on Metal");
        };
        let idf_buf = idf_metal.buffer().clone();
        assert_eq!(idf_layout.start_offset(), 0);

        // Allocate the down destination buffer `[num_tokens*top_k, 1, hidden]`,
        // which is logically equivalent to `[num_tokens, top_k, hidden]`.
        let dn_out_elems = num_tokens * top_k * hidden;
        let dn_dst_buf = metal_device.new_buffer(
            dn_out_elems, DType::F32, "moe_fused_down_dst",
        )?;

        {
            let encoder = metal_device.command_encoder()?;
            moe_kernel::call_quantized_matmul_mv_id_t(
                metal_device.device(),
                &encoder,
                metal_device.kernels(),
                self.fused_down_dtype,
                moe_kernel::MoeDispatchShape {
                    num_experts: self.num_experts,
                    n: hidden,
                    k: intermediate,
                    n_tokens: num_tokens * top_k,
                    n_expert_used: 1,
                },
                dn_metal.buffer(),
                &sw_buf,
                sw_offset,
                &idf_buf,
                &dn_dst_buf,
                0,
            ).map_err(|e| anyhow::anyhow!("fused down dispatch: {e:?}"))?;
        }
        self.counters.dispatches_per_token.fetch_add(1, Ordering::Relaxed);

        let dn_storage = candle_core::MetalStorage::new(
            dn_dst_buf,
            metal_device.clone(),
            dn_out_elems,
            DType::F32,
        );
        let down_out = Tensor::from_storage(
            Storage::Metal(dn_storage),
            (num_tokens, top_k, hidden),
            candle_core::op::BackpropOp::none(),
            false,
        );

        // --- GPU per_expert_scale gather via top_k_indices ---------------
        // `per_expert_scale` is `[num_experts]` f32 on device. We want
        // `gathered_scale[i, k] = per_expert_scale[top_k_indices[i, k]]`,
        // shape `[num_tokens, top_k]`.
        let top_k_indices_flat = top_k_indices.reshape((num_tokens * top_k,))?;
        let per_expert_scale_f32 = self.per_expert_scale.to_dtype(DType::F32)?;
        let gathered_flat = per_expert_scale_f32.index_select(&top_k_indices_flat, 0)?;
        let gathered_scale = gathered_flat.reshape((num_tokens, top_k))?;
        self.counters.dispatches_per_token.fetch_add(4, Ordering::Relaxed);

        // --- Combine softmax top-k weights with per-expert scale ---------
        // Both `top_k_weights` and `gathered_scale` are f32 `[num_tokens, top_k]`.
        let w_total = (top_k_weights * gathered_scale)?; // [num_tokens, top_k]
        // Broadcast-ready shape for per-hidden multiply.
        let w_total_3d = w_total.unsqueeze(D::Minus1)?; // [num_tokens, top_k, 1]
        let weighted = down_out.broadcast_mul(&w_total_3d)?; // [num_tokens, top_k, hidden]
        self.counters.dispatches_per_token.fetch_add(3, Ordering::Relaxed);

        // --- Sum over top_k → [num_tokens, hidden] → reshape to output -----
        let summed = weighted.sum(1)?; // [num_tokens, hidden]
        let out = summed
            .reshape((b_sz, seq_len, hidden))?
            .to_dtype(out_dtype)?;
        self.counters.dispatches_per_token.fetch_add(3, Ordering::Relaxed);

        Ok(out)
    }
}

// ---------------------------------------------------------------------------
// Decoder Layer
// ---------------------------------------------------------------------------

struct DecoderLayer {
    self_attn: Attention,
    // Dense MLP
    mlp: Mlp,
    // MoE
    moe: MoeBlock,
    // Norms
    input_layernorm: RmsNorm,
    post_attention_layernorm: RmsNorm,
    pre_feedforward_layernorm: RmsNorm,
    post_feedforward_layernorm: RmsNorm,
    pre_feedforward_layernorm_2: RmsNorm,
    post_feedforward_layernorm_1: RmsNorm,
    post_feedforward_layernorm_2: RmsNorm,
    layer_scalar: Tensor,
    counters: Arc<DispatchCounters>,
}

impl DecoderLayer {
    fn forward(
        &mut self,
        xs: &Tensor,
        seqlen_offset: usize,
    ) -> Result<Tensor> {
        // 1. Attention block
        let normed = self.input_layernorm.forward(xs)?;
        let attn_out = self.self_attn.forward(&normed, seqlen_offset)?;
        let attn_out = self.post_attention_layernorm.forward(&attn_out)?;
        // ADR-005 1bNEW.0b — Walk Exception unwind.
        //
        // Previously this was a single fused call (Phase 1 1b.10):
        //   let (normed, xs) = self.pre_feedforward_layernorm
        //       .forward_with_residual(&attn_out, xs)?;
        // which computed `sum = xs + attn_out` and `normed = norm(sum)`
        // inside `RmsNorm::forward_with_residual`. References:
        //   - mlx-lm gemma4_text.py:339-340
        //       h = self.post_attention_layernorm(h)
        //       h = residual + h
        //   - llama.cpp src/models/gemma4-iswa.cpp:117-122
        //       cur = build_norm(cur, model.layers[il].attn_post_norm, ...);
        //       ggml_tensor * attn_out = ggml_add(ctx0, cur, inpL);
        // Both references apply the post-attention norm, then the residual
        // add as separate ops. `pre_feedforward_layernorm` is in a different
        // position in the graph (mlx-lm line 345, llama.cpp separate norm
        // call later) but the immediate pattern here — ADD residual then
        // NORM — is faithful to both. We restore Walk fidelity now; any
        // re-fusion is Run territory (ADR-005 Anti-Goal #14 + Walk Exception
        // Register).
        let xs = (xs + &attn_out)?;
        // The explicit add is one candle op; the RmsNorm::forward call below
        // self-counts its 11 ops into both `dispatches_per_token` and
        // `norm_dispatches_per_token`.
        self.counters.dispatches_per_token.fetch_add(1, Ordering::Relaxed);
        let normed = self.pre_feedforward_layernorm.forward(&xs)?;

        // 2. Dense MLP and MoE run in PARALLEL from the same residual
        let residual = &xs;

        // Dense MLP branch (uses the already-normed input from the fused op above)
        let mlp_out = self.mlp.forward(&normed)?;
        let mlp_normed = self.post_feedforward_layernorm_1.forward(&mlp_out)?;

        // MoE branch (router takes raw residual; experts take pre-normed residual)
        let normed_moe = self.pre_feedforward_layernorm_2.forward(&xs)?;
        let moe_out = self.moe.forward(&normed_moe, &xs)?;
        let moe_normed = self.post_feedforward_layernorm_2.forward(&moe_out)?;

        // Sum MLP and MoE outputs, apply final post-FFW norm.
        //
        // ADR-005 1bNEW.4: this is the single NORM→ADD site per layer
        // that maps to the F=3 fused kernel variant. In `fused` mode
        // `forward_with_post_residual` dispatches the kernel once for
        // the `(combined*scale)*w + residual` pattern; in `loop` mode
        // it falls back to the explicit two-op sequence
        // `t = norm(combined); xs = residual + t` — same as the
        // pre-1bNEW.4 inlined ops. This is NOT a re-fuse of
        // `forward_with_residual` (1bNEW.0b Walk Exception unwind);
        // that site is the *pre-attention-norm* residual add which
        // stays ADD-THEN-NORM. This site is the post-FFW combiner,
        // which is NORM-THEN-ADD in both mlx-lm and llama.cpp
        // references — see `forward_with_post_residual` docstring for
        // the citation walk.
        let combined = (mlp_normed + moe_normed)?;
        let xs = self
            .post_feedforward_layernorm
            .forward_with_post_residual(&combined, residual)?;

        // 3. Layer scalar
        let out = xs.broadcast_mul(&self.layer_scalar)?;
        // Outside-self-count ops:
        //   - 1 mlp+moe add (the `(mlp_normed + moe_normed)` above).
        //   - 1 broadcast_mul (layer_scalar).
        //   - `forward_with_post_residual` self-counts its own dispatches
        //     (1 for the fused F=3 path, 12 for the loop path — self +
        //     explicit residual add).
        // Total outside sub-modules: 2.
        self.counters.dispatches_per_token.fetch_add(2, Ordering::Relaxed);
        Ok(out)
    }

    #[allow(dead_code)]
    fn clear_cache(&mut self) {
        self.self_attn.clear_cache();
    }
}

// ---------------------------------------------------------------------------
// Full Model
// ---------------------------------------------------------------------------

pub struct Gemma4Model {
    embed_tokens: Embedding,
    layers: Vec<DecoderLayer>,
    norm: RmsNorm,
    lm_head_weight: Tensor,
    /// ADR-005 1bNEW.17 — parallel F16 copy of `lm_head_weight`.
    /// `None` under `--lm-head-kernel=loop`; `Some(f16_tensor)` under
    /// `--lm-head-kernel=fused`, where `Gemma4Model::forward` routes
    /// the final vocab projection through
    /// `lm_head_kernel::lm_head_forward_fused` instead of the dense
    /// F32 `lm_head_weight.matmul(.t())` path. The F32 copy is
    /// retained alongside so `loop` mode stays byte-identical (same
    /// pattern as `--moe-kernel=loop` keeping the per-expert
    /// `QMatMul::forward` vec).
    lm_head_f16_weight: Option<Tensor>,
    /// ADR-005 1bNEW.17 — which of the two paths the forward pass
    /// should take. Resolved at model load time from the CLI flag.
    lm_head_mode: LmHeadKernelModeImpl,
    hidden_size: usize,
    final_logit_softcapping: Option<f64>,
    device: Device,
    counters: Arc<DispatchCounters>,
    /// ADR-005 1bNEW.4 — retained handle to the runtime-compiled
    /// RmsNorm library + pipelines. Every `RmsNorm` in the model
    /// carries a clone of this bundle; holding the master copy on the
    /// model ensures the `Arc<RmsNormPipelines>` strong count never
    /// drops to zero while a forward pass is in flight.
    #[allow(dead_code)]
    rms_kernel: RmsNormKernel,
}

impl Gemma4Model {
    /// Shared handle to the dispatch counters so callers (benchmark harness,
    /// sampler) can reset and snapshot them.
    pub fn counters(&self) -> Arc<DispatchCounters> {
        self.counters.clone()
    }
}

impl Gemma4Model {
    /// Load model from GGUF + config. Backward-compatible entry point
    /// that defaults to the Phase-1 `Loop` MoE path for every layer.
    /// Kept for existing call sites (tests, older bins) that pre-date
    /// the ADR-005 1bNEW.1 `--moe-kernel` flag.
    #[allow(dead_code)]
    pub fn load(cfg: &Gemma4Config, gguf: &GgufModel, device: &Device) -> Result<Self> {
        Self::load_with_modes(
            cfg,
            gguf,
            device,
            MoeKernelMode::Loop,
            RmsKernelMode::Loop,
            RopeKernelModeImpl::Loop,
            LmHeadKernelModeImpl::Loop,
            KvCacheKernelMode::SliceScatter,
        )
    }

    /// Back-compat wrapper retained so existing callers (tests, older
    /// bin entry points) keep working without knowing about the
    /// `--rms-norm-kernel` knob. Equivalent to `load_with_modes(.., Loop)`.
    #[allow(dead_code)]
    pub fn load_with_moe_mode(
        cfg: &Gemma4Config,
        gguf: &GgufModel,
        device: &Device,
        moe_mode: MoeKernelMode,
    ) -> Result<Self> {
        Self::load_with_modes(
            cfg,
            gguf,
            device,
            moe_mode,
            RmsKernelMode::Loop,
            RopeKernelModeImpl::Loop,
            LmHeadKernelModeImpl::Loop,
            KvCacheKernelMode::SliceScatter,
        )
    }

    /// Load model from GGUF + config with explicit MoE and RmsNorm
    /// dispatch modes.
    ///
    /// ADR-005 1bNEW.1 Phase C widened `MoeKernelMode::Fused` to every
    /// layer. ADR-005 1bNEW.4 Phase B adds `RmsNormKernelMode::Fused`,
    /// which, when set, compiles the downstream MSL library once via
    /// `rms_norm_kernel::RmsNormPipelines::new(metal_device)` and clones
    /// the resulting `Arc` into every `RmsNorm`, `Attention`, and
    /// `MoeBlock` sub-structure. `Loop` mode leaves the pipelines
    /// field `None` and pays no compile cost, matching the pre-1bNEW.4
    /// baseline byte-for-byte under Layer A token match.
    pub fn load_with_modes(
        cfg: &Gemma4Config,
        gguf: &GgufModel,
        device: &Device,
        moe_mode: MoeKernelMode,
        rms_norm_mode: RmsKernelMode,
        rope_mode: RopeKernelModeImpl,
        lm_head_mode: LmHeadKernelModeImpl,
        kv_cache_mode: KvCacheKernelMode,
    ) -> Result<Self> {
        // ADR-005 1bNEW.0: shared dispatch counters. Every sub-structure that
        // issues candle ops holds a clone of this Arc.
        let counters = DispatchCounters::new();

        // ADR-005 1bNEW.4 — compile the fused RmsNorm library at model
        // load time so every forward pass sees a warm PSO (no
        // first-call spike, matching 1bNEW.12's warmup discipline).
        // Under `Loop` mode we skip the compile entirely — the
        // `RmsNormKernel` bundle carries `pipelines = None` and the
        // per-site `is_fused()` check short-circuits to the 11-op
        // manual chain unchanged.
        let rms_kernel = match rms_norm_mode {
            RmsKernelMode::Loop => RmsNormKernel::loop_mode(),
            RmsKernelMode::Fused => match device {
                Device::Metal(md) => RmsNormKernel::fused_mode(md)?,
                other => anyhow::bail!(
                    "--rms-norm-kernel=fused requires a Metal device, got {other:?}"
                ),
            },
        };
        tracing::info!("RmsNorm dispatch mode: {:?}", rms_norm_mode);

        // ADR-005 1bNEW.6 — compile the fused RoPE library at model
        // load time so every forward pass sees a warm PSO (no first-
        // call spike, matching 1bNEW.4's RmsNorm and 1bNEW.12's
        // warmup discipline). `Loop` mode skips the compile and
        // leaves `pipelines = None`, matching pre-1bNEW.6 exactly.
        let rope_kernel = match rope_mode {
            RopeKernelModeImpl::Loop => RopeKernel::loop_mode(),
            RopeKernelModeImpl::Fused => match device {
                Device::Metal(md) => RopeKernel::fused_mode(md)?,
                other => anyhow::bail!(
                    "--rope-kernel=fused requires a Metal device, got {other:?}"
                ),
            },
        };
        tracing::info!("RoPE dispatch mode: {:?}", rope_mode);
        // ADR-005 1bNEW.20 — KV cache append mode is a pure selector
        // (no kernel compile), propagated to every per-layer `KvCache`
        // at construction time below.
        tracing::info!("KV cache append mode: {:?}", kv_cache_mode);

        // Embedding — GGUF stores as [hidden, vocab]; Candle Embedding wants [vocab, hidden]
        let embed_w = gguf.get_tensor("token_embd.weight", MODEL_DTYPE)?;
        let embed_w = if embed_w.dim(0)? == cfg.hidden_size && embed_w.dim(1)? == cfg.vocab_size {
            embed_w.t()?.contiguous()?
        } else {
            embed_w
        };
        let embed = Embedding::new(embed_w.clone(), cfg.hidden_size);

        // Rotary embeddings (shared across same-type layers).
        // Cap max seq len for RoPE tables to something reasonable for startup.
        //
        // **ADR-005 1bNEW.18 (2026-04-11):** load `rope_freqs.weight`
        // from the GGUF and attach it to the global `RotaryEmbedding`
        // as a per-pair frequency mask. llama.cpp uses this tensor
        // on non-SWA (full-attention) layers only — see
        // `/opt/llama.cpp/src/models/gemma4-iswa.cpp:55-59`:
        //   `ggml_tensor * freq_factors = nullptr;`
        //   `if (!hparams.is_swa(il)) {`
        //   `    freq_factors = model.layers[il].rope_freqs;`
        //   `}`
        // The tensor is a global-scope (no `blk.N` prefix) F32 tensor
        // of shape `[head_dim/2]`, loaded in llama.cpp at
        // `/opt/llama.cpp/src/llama-model.cpp:4311-4313`. For Gemma 4
        // 26B MoE it's `[256]` with the pattern
        // `[1.0]×64 + [1e+30]×192` — the first 64 pairs rotate at
        // their natural angles, the remaining 192 rotate to identity
        // because `theta / 1e+30 → 0 → cos=1, sin=0`. Sliding layers
        // pass `freq_factors = None`, preserving their pre-1bNEW.18
        // sin/cos tables byte-identically.
        //
        // We load via `get_tensor("rope_freqs.weight", DType::F32)`
        // and immediately move to host as a `Vec<f32>` so the `build`
        // helper can fold the mask into `inv_freq` for the loop path
        // AND keep a `[half]` device tensor for fused-path `src2`
        // binding. See `docs/spike-C-results.md` Parts 3-5 for the
        // full end-to-end root cause analysis that led to this
        // change.
        let max_rope_len = cfg.max_position_embeddings.min(8192);

        let rope_freqs_host: Vec<f32> = {
            let t = gguf
                .get_tensor("rope_freqs.weight", DType::F32)
                .map_err(|e| {
                    anyhow::anyhow!(
                        "ADR-005 1bNEW.18: failed to load `rope_freqs.weight` \
                         from GGUF — required for Gemma 4 global-layer RoPE \
                         per /opt/llama.cpp/src/models/gemma4-iswa.cpp:55-59: {e}"
                    )
                })?;
            let t = t.to_device(&Device::Cpu)?.flatten_all()?;
            t.to_vec1::<f32>()?
        };
        let expected_half = cfg.global_head_dim / 2;
        if rope_freqs_host.len() != expected_half {
            anyhow::bail!(
                "rope_freqs.weight has {} elements but global_head_dim/2 = {}",
                rope_freqs_host.len(),
                expected_half,
            );
        }
        tracing::info!(
            "rope_freqs.weight: {} elements, ones@[0..{}), 1e+30@[{}..{})",
            rope_freqs_host.len(),
            rope_freqs_host.iter().take_while(|v| **v < 1e20).count(),
            rope_freqs_host.iter().take_while(|v| **v < 1e20).count(),
            rope_freqs_host.len(),
        );

        let rope_sliding = Arc::new(RotaryEmbedding::new(
            cfg.head_dim,
            max_rope_len,
            cfg.rope_theta_sliding,
            None, // sliding layers do not use freq_factors
            device,
            rope_kernel.clone(),
            counters.clone(),
        )?);
        let rope_global = Arc::new(RotaryEmbedding::new(
            cfg.global_head_dim,
            max_rope_len,
            cfg.rope_theta_global,
            Some(rope_freqs_host),
            device,
            rope_kernel.clone(),
            counters.clone(),
        )?);

        // Layers
        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        for i in 0..cfg.num_hidden_layers {
            eprint!("\r  Loading layer {}/{}...", i + 1, cfg.num_hidden_layers);
            let lp = format!("blk.{}", i);
            let is_full = cfg.is_full_attention(i);
            let head_dim = cfg.head_dim_for_layer(i);
            let num_kv_heads = cfg.num_kv_heads_for_layer(i);

            let rotary = if is_full { rope_global.clone() } else { rope_sliding.clone() };

            // Attention
            let k_eq_v = is_full && cfg.attention_k_eq_v;
            let k_proj = load_qlinear(gguf, &format!("{}.attn_k", lp), &counters)?;
            let v_proj = if k_eq_v {
                // V is tied to K — create a dummy (never used in forward)
                QLinear::new(k_proj.inner.clone(), k_proj.bias.clone(), counters.clone())
            } else {
                load_qlinear(gguf, &format!("{}.attn_v", lp), &counters)?
            };

            let attn = Attention {
                q_proj: load_qlinear(gguf, &format!("{}.attn_q", lp), &counters)?,
                k_proj,
                v_proj,
                o_proj: load_qlinear(gguf, &format!("{}.attn_output", lp), &counters)?,
                q_norm: load_rms_norm(gguf, &format!("{}.attn_q_norm", lp), cfg.rms_norm_eps, &counters, &rms_kernel)?,
                k_norm: load_rms_norm(gguf, &format!("{}.attn_k_norm", lp), cfg.rms_norm_eps, &counters, &rms_kernel)?,
                num_heads: cfg.num_attention_heads,
                num_kv_heads,
                head_dim,
                rotary_emb: rotary,
                kv_cache: KvCache::new(
                    num_kv_heads, head_dim, device,
                    if is_full { None } else { Some(cfg.sliding_window) },
                    kv_cache_mode,
                )?,
                k_eq_v,
                counters: counters.clone(),
                rms_kernel: rms_kernel.clone(),
            };

            // Dense MLP
            let mlp = Mlp {
                gate_proj: load_qlinear(gguf, &format!("{}.ffn_gate", lp), &counters)?,
                up_proj: load_qlinear(gguf, &format!("{}.ffn_up", lp), &counters)?,
                down_proj: load_qlinear(gguf, &format!("{}.ffn_down", lp), &counters)?,
                counters: counters.clone(),
            };

            // MoE: load per-expert QMatMul from byte-sliced 3D QTensor.
            // GGUF stores expert weights as 3D: candle reads dims as [num_experts, out, in].
            // We extract each expert's raw quantized bytes and construct a 2D QMatMul.
            let gu_qt = gguf.get_qtensor(&format!("{}.ffn_gate_up_exps.weight", lp))?;
            let dn_qt = gguf.get_qtensor(&format!("{}.ffn_down_exps.weight", lp))?;

            if i == 0 {
                tracing::info!(
                    "Expert gate_up: {:?} {:?}, down: {:?} {:?}",
                    gu_qt.shape(), gu_qt.dtype(), dn_qt.shape(), dn_qt.dtype()
                );
            }

            let gu_data = gu_qt.data()?;
            let dn_data = dn_qt.data()?;
            let gu_dtype = gu_qt.dtype();
            let dn_dtype = dn_qt.dtype();
            let gu_shape = gu_qt.shape();
            let dn_shape = dn_qt.shape();
            let gu_bytes_per_expert = gu_data.len() / cfg.num_experts;
            let dn_bytes_per_expert = dn_data.len() / cfg.num_experts;
            // Per-expert 2D shape: [out_features, in_features]
            let gu_expert_shape = (gu_shape.dims()[1], gu_shape.dims()[2]);
            let dn_expert_shape = (dn_shape.dims()[1], dn_shape.dims()[2]);

            let mut expert_gate_up = Vec::with_capacity(cfg.num_experts);
            let mut expert_down = Vec::with_capacity(cfg.num_experts);
            for e in 0..cfg.num_experts {
                let gu_slice = &gu_data[e * gu_bytes_per_expert..(e + 1) * gu_bytes_per_expert];
                let gu_storage = candle_core::quantized::QStorage::from_data(
                    std::borrow::Cow::Borrowed(gu_slice), device, gu_dtype,
                )?;
                let gu_qtensor = Arc::new(candle_core::quantized::QTensor::new(
                    gu_storage, gu_expert_shape,
                )?);
                expert_gate_up.push(QMatMul::from_arc(gu_qtensor)?);

                let dn_slice = &dn_data[e * dn_bytes_per_expert..(e + 1) * dn_bytes_per_expert];
                let dn_storage = candle_core::quantized::QStorage::from_data(
                    std::borrow::Cow::Borrowed(dn_slice), device, dn_dtype,
                )?;
                let dn_qtensor = Arc::new(candle_core::quantized::QTensor::new(
                    dn_storage, dn_expert_shape,
                )?);
                expert_down.push(QMatMul::from_arc(dn_qtensor)?);
            }

            // ADR-005 1bNEW.1 Phase C: every layer runs the fused path.
            //
            // Phase B gated on `i == 0` to bisect a single-layer change;
            // with Phase B token-matched bit-exact (top-1 preserved,
            // top-10 Δ ≤ 1.5e-5), Phase C widens the gate to every layer.
            // The per-layer `Fused` branch is a trivial identity map now,
            // but the match is kept so future phases (e.g. skip global
            // attention layers, conditional fusion) can revert to a
            // per-layer decision without changing the call sites.
            let layer_mode = match moe_mode {
                MoeKernelMode::Loop => MoeKernelMode::Loop,
                MoeKernelMode::Fused => MoeKernelMode::Fused,
            };

            // Optionally construct the 3D fused weight storages. We only
            // pay the allocation when this layer is actually running in
            // `Fused` mode — keeps the `Loop` baseline byte-identical.
            #[cfg(feature = "metal")]
            let (fused_gate_up, fused_down, fused_gate_up_dtype, fused_down_dtype) = if layer_mode == MoeKernelMode::Fused {
                let gu_3d = candle_core::quantized::QStorage::from_data(
                    std::borrow::Cow::Borrowed(&gu_data), device, gu_dtype,
                )?;
                let dn_3d = candle_core::quantized::QStorage::from_data(
                    std::borrow::Cow::Borrowed(&dn_data), device, dn_dtype,
                )?;
                // Map candle_core::quantized::GgmlDType → the
                // candle_metal_kernels tag the fused wrapper uses.
                let gu_tag = core_to_metal_ggml(gu_dtype)?;
                let dn_tag = core_to_metal_ggml(dn_dtype)?;
                (Some(gu_3d), Some(dn_3d), gu_tag, dn_tag)
            } else {
                // Placeholder tags — never read when mode is Loop.
                (None, None, MetalGgmlDType::F32, MetalGgmlDType::F32)
            };

            let moe = MoeBlock {
                router_proj: load_qlinear(gguf, &format!("{}.ffn_gate_inp", lp), &counters)?,
                router_scale: gguf.get_tensor(&format!("{}.ffn_gate_inp.scale", lp), MODEL_DTYPE)?,
                per_expert_scale: gguf.get_tensor(&format!("{}.ffn_down_exps.scale", lp), MODEL_DTYPE)?,
                per_expert_scale_cpu: gguf.get_tensor(&format!("{}.ffn_down_exps.scale", lp), MODEL_DTYPE)?
                    .to_dtype(DType::F32)?.to_vec1::<f32>()?,
                expert_gate_up,
                expert_down,
                num_experts: cfg.num_experts,
                top_k: cfg.top_k_experts,
                moe_intermediate_size: cfg.moe_intermediate_size,
                hidden_size: cfg.hidden_size,
                counters: counters.clone(),
                #[cfg(feature = "metal")]
                fused_gate_up,
                #[cfg(feature = "metal")]
                fused_down,
                #[cfg(feature = "metal")]
                fused_gate_up_dtype,
                #[cfg(feature = "metal")]
                fused_down_dtype,
                mode: layer_mode,
                rms_kernel: rms_kernel.clone(),
            };

            let layer = DecoderLayer {
                self_attn: attn,
                mlp,
                moe,
                input_layernorm: load_rms_norm(gguf, &format!("{}.attn_norm", lp), cfg.rms_norm_eps, &counters, &rms_kernel)?,
                post_attention_layernorm: load_rms_norm(gguf, &format!("{}.post_attention_norm", lp), cfg.rms_norm_eps, &counters, &rms_kernel)?,
                pre_feedforward_layernorm: load_rms_norm(gguf, &format!("{}.ffn_norm", lp), cfg.rms_norm_eps, &counters, &rms_kernel)?,
                post_feedforward_layernorm: load_rms_norm(gguf, &format!("{}.post_ffw_norm", lp), cfg.rms_norm_eps, &counters, &rms_kernel)?,
                pre_feedforward_layernorm_2: load_rms_norm(gguf, &format!("{}.pre_ffw_norm_2", lp), cfg.rms_norm_eps, &counters, &rms_kernel)?,
                post_feedforward_layernorm_1: load_rms_norm(gguf, &format!("{}.post_ffw_norm_1", lp), cfg.rms_norm_eps, &counters, &rms_kernel)?,
                post_feedforward_layernorm_2: load_rms_norm(gguf, &format!("{}.post_ffw_norm_2", lp), cfg.rms_norm_eps, &counters, &rms_kernel)?,
                layer_scalar: gguf.get_tensor(&format!("{}.layer_output_scale.weight", lp), MODEL_DTYPE)?,
                counters: counters.clone(),
            };

            layers.push(layer);
        }
        eprintln!("\r  Loaded {}/{} layers.    ", cfg.num_hidden_layers, cfg.num_hidden_layers);

        // Final norm
        let norm = load_rms_norm(gguf, "output_norm", cfg.rms_norm_eps, &counters, &rms_kernel)?;

        // lm_head is tied to embed_tokens
        let lm_head_weight = embed_w;

        // ADR-005 1bNEW.17 — parallel F16 copy of the tied lm_head
        // weight for the fused path. Under `Loop` mode we pay no
        // extra allocation; under `Fused` we allocate a 1.48 GB F16
        // tensor alongside the existing 2.95 GB F32 `lm_head_weight`
        // copy, so `--lm-head-kernel=loop` remains byte-identical to
        // pre-1bNEW.17 and bisect-safe.
        //
        // The cast is `embed_w.to_dtype(DType::F16)`, which runs a
        // single F32→F16 conversion pass on the device. The source
        // `embed_w` is already on the target device at this point
        // (it's the Embedding's weight, loaded at `gemma4.rs:1636`).
        // `token_embd.weight` is physically F16 in the Gemma 4 DWQ
        // GGUF (`n_bytes=1_476_395_008`, verified via gguf.GGUFReader),
        // so the F16 Metal tensor bit-matches the original GGUF
        // storage modulo the F32-round-trip through the existing
        // `get_tensor` dequant path. A later item can thread the F16
        // load directly from GGUF without the F32 intermediate; for
        // Phase A/B/C the one-time load-time cast is the simpler
        // bisect-safe change.
        let lm_head_f16_weight: Option<Tensor> = match lm_head_mode {
            LmHeadKernelModeImpl::Loop => None,
            LmHeadKernelModeImpl::Fused => {
                tracing::info!(
                    "lm_head_kernel: allocating F16 copy of token_embd.weight \
                     ({} × {} = {:.2} GB)",
                    cfg.vocab_size,
                    cfg.hidden_size,
                    (cfg.vocab_size as f64 * cfg.hidden_size as f64 * 2.0) / 1e9,
                );
                Some(lm_head_weight.to_dtype(DType::F16)?)
            }
        };
        tracing::info!("lm_head dispatch mode: {:?}", lm_head_mode);

        Ok(Self {
            embed_tokens: embed,
            layers,
            norm,
            lm_head_weight,
            lm_head_f16_weight,
            lm_head_mode,
            hidden_size: cfg.hidden_size,
            final_logit_softcapping: cfg.final_logit_softcapping,
            device: device.clone(),
            counters,
            rms_kernel,
        })
    }

    /// Forward pass: [batch, seq_len] → logits [batch, 1, vocab_size].
    pub fn forward(&mut self, input_ids: &Tensor, seqlen_offset: usize) -> Result<Tensor> {
        // ADR-005 1bNEW.0: count every forward call so per-token averages
        // divide correctly.
        self.counters.forward_count.fetch_add(1, Ordering::Relaxed);

        let (_b_sz, seq_len) = input_ids.dims2()?;

        // Embed and scale (embedding lookup + scalar mul = 2 ops)
        let mut xs = self.embed_tokens.forward(input_ids)?;
        xs = (xs * (self.hidden_size as f64).sqrt())?;
        self.counters.dispatches_per_token.fetch_add(2, Ordering::Relaxed);

        // ADR-005 1bNEW.10: no mask buffer is built for prefill — candle's SDPA
        // full kernel handles causal masking internally via `do_causal=true` inside
        // `Attention::forward`. Decode (`seq_len == 1`) never needs a mask.

        // Transformer layers
        for (_i, layer) in self.layers.iter_mut().enumerate() {
            xs = layer.forward(&xs, seqlen_offset)?;
        }

        // Final norm + lm_head (last token only)
        let last_hidden = xs.narrow(1, seq_len - 1, 1)?;
        let normed = self.norm.forward(&last_hidden)?;
        // lm_head is tied to embed_tokens which is [vocab, hidden]
        // logits = normed @ lm_head.T  →  [1,1,hidden] @ [hidden, vocab]
        let normed_2d = normed.reshape((1, self.hidden_size))?;
        // ADR-005 1bNEW.17 Phase B — branch on `--lm-head-kernel`.
        //
        // Loop (Phase-1 baseline): dense F32 matmul against the 2.95 GB
        //   dequantized `token_embd.weight` copy, reading 2.95 GB/token.
        //   2 dispatches: `t()` (view-only in candle, but accounted for
        //   bookkeeping parity) + `matmul`. Counted as 2 ops.
        //
        // Fused (ADR-005 1bNEW.17): native F16 gemm against the 1.48 GB
        //   F16 copy, reading 1.48 GB/token. Via
        //   `lm_head_kernel::lm_head_forward_fused` which is `to_dtype(F16)`
        //   + `t()` + `matmul` + `to_dtype(F32)` = 4 candle ops. The 2
        //   extra ops are negligible (cast on [1, 2816] and [1, 262144]
        //   tensors, no weight reads). Counted as 4 ops so metrics.txt
        //   stays honest. The matmul kernel is candle's `call_mlx_gemm`
        //   GemmDType::F16 path at `candle-core/src/metal_backend/mod.rs:1695-1709`.
        let (logits, lm_head_ops): (Tensor, u64) = match self.lm_head_mode {
            LmHeadKernelModeImpl::Loop => {
                let logits = normed_2d.matmul(&self.lm_head_weight.t()?)?;
                (logits, 2)
            }
            LmHeadKernelModeImpl::Fused => {
                // `lm_head_f16_weight` is populated at load time when
                // `lm_head_mode == Fused`; unwrapping here is correct by
                // construction and panicking on `None` would indicate a
                // load-path bug (Anti-Goal #7 — fail loud, no silent
                // fallback).
                let w_f16 = self.lm_head_f16_weight.as_ref().ok_or_else(|| {
                    anyhow::anyhow!(
                        "1bNEW.17: lm_head_mode == Fused but lm_head_f16_weight is None"
                    )
                })?;
                let logits = lm_head_kernel::lm_head_forward_fused(&normed_2d, w_f16)?;
                (logits, 4)
            }
        };
        let logits = logits.unsqueeze(0)?; // [1, 1, vocab]
        // narrow + reshape + (lm_head branch) + unsqueeze ops (norm self-counts).
        // Phase-1 accounting kept 5 ops (narrow+reshape+matmul+t+unsqueeze =
        // 5). The `Loop` branch here issues the same 5. The `Fused` branch
        // issues 7 (narrow+reshape+to_f16+matmul+t+to_f32+unsqueeze). The
        // +2 is bookkeeping-only — no weight traffic, no new kernels.
        self.counters
            .dispatches_per_token
            .fetch_add(3 + lm_head_ops, Ordering::Relaxed);

        // Softcapping
        match self.final_logit_softcapping {
            Some(sc) => {
                let out = ((logits / sc)?.tanh()? * sc)?;
                // div, tanh, mul = 3 ops.
                self.counters.dispatches_per_token.fetch_add(3, Ordering::Relaxed);
                Ok(out)
            }
            None => Ok(logits),
        }
    }

    #[allow(dead_code)]
    pub fn clear_kv_cache(&mut self) {
        for layer in self.layers.iter_mut() {
            layer.clear_cache();
        }
    }

    #[allow(dead_code)]
    fn has_nan(t: &Tensor) -> bool {
        t.to_dtype(DType::F32)
            .and_then(|t| t.flatten_all())
            .and_then(|t| t.to_vec1::<f32>())
            .map(|v| v.iter().any(|x| x.is_nan()))
            .unwrap_or(true)
    }

    #[allow(dead_code)]
    fn nan_check(t: &Tensor, label: &str) -> Result<()> {
        let flat = t.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
        let nan_count = flat.iter().filter(|v| v.is_nan()).count();
        let inf_count = flat.iter().filter(|v| v.is_infinite()).count();
        let sample: Vec<f32> = flat.iter().take(5).copied().collect();
        eprintln!("[NAN] {}: shape={:?} nan={}/{} inf={} sample={:?}",
            label, t.shape(), nan_count, flat.len(), inf_count, sample);
        Ok(())
    }
}

#[cfg(test)]
mod kv_cache_in_place_tests {
    //! ADR-005 1bNEW.20 — KV cache in-place append unit tests.
    //!
    //! Phase A gate: the `InPlace` path must produce element-wise identical
    //! K/V tensors to the `SliceScatter` baseline for every (shape, append
    //! sequence, sliding-window) combination that the live forward path can
    //! hit. The math is identical — only the op sequence changes — so the
    //! assertion is strict `max |Δ| == 0` with zero tolerance.
    //!
    //! These tests deliberately do NOT need a GGUF — they construct
    //! `KvCache::new` directly and call `append` with synthetic Tensors.
    //! That keeps them fast (<1 s each) and lets them run in the normal
    //! `cargo test` sweep (no `#[ignore]`).

    use super::{KvCache, KvCacheKernelMode};
    use candle_core::{DType, Device, Tensor};

    /// Maximum elementwise absolute difference between two tensors,
    /// as f64 for headroom.
    fn max_abs_diff_strict(a: &Tensor, b: &Tensor) -> f64 {
        let a_flat: Vec<f32> = a.to_dtype(DType::F32).unwrap().flatten_all().unwrap().to_vec1().unwrap();
        let b_flat: Vec<f32> = b.to_dtype(DType::F32).unwrap().flatten_all().unwrap().to_vec1().unwrap();
        assert_eq!(a_flat.len(), b_flat.len(), "shape mismatch");
        a_flat
            .iter()
            .zip(b_flat.iter())
            .map(|(x, y)| (*x as f64 - *y as f64).abs())
            .fold(0.0_f64, f64::max)
    }

    /// Build a deterministic float tensor shaped `[1, num_kv_heads, new_len, head_dim]`.
    /// Uses a simple linear ramp so each element has a unique, identifiable
    /// value — if a stride is wrong we'll see it immediately.
    fn synthetic_kv(
        num_kv_heads: usize,
        new_len: usize,
        head_dim: usize,
        offset: usize,
        device: &Device,
    ) -> Tensor {
        let total = num_kv_heads * new_len * head_dim;
        let data: Vec<f32> = (0..total).map(|i| (offset + i) as f32 * 1e-3).collect();
        Tensor::from_vec(data, (1, num_kv_heads, new_len, head_dim), device).unwrap()
    }

    /// Core equivalence test: append the same sequence of K/V tensors
    /// through both `SliceScatter` and `InPlace` and assert the active
    /// view matches bit-for-bit at every step.
    fn run_kv_equivalence(
        num_kv_heads: usize,
        head_dim: usize,
        sliding_window: Option<usize>,
        append_lengths: &[usize],
    ) -> (Vec<f64>, Vec<f64>) {
        let device = Device::new_metal(0).expect("Metal device required");

        let mut cache_ss = KvCache::new(
            num_kv_heads,
            head_dim,
            &device,
            sliding_window,
            KvCacheKernelMode::SliceScatter,
        )
        .unwrap();
        let mut cache_ip = KvCache::new(
            num_kv_heads,
            head_dim,
            &device,
            sliding_window,
            KvCacheKernelMode::InPlace,
        )
        .unwrap();

        let mut k_diffs = Vec::with_capacity(append_lengths.len());
        let mut v_diffs = Vec::with_capacity(append_lengths.len());

        let mut offset = 0;
        for (step, &new_len) in append_lengths.iter().enumerate() {
            // Distinct synthetic K and V (offset by a large constant so
            // we catch any mix-up of the two buffers).
            let k_new = synthetic_kv(num_kv_heads, new_len, head_dim, offset, &device);
            offset += num_kv_heads * new_len * head_dim;
            let v_new = synthetic_kv(num_kv_heads, new_len, head_dim, offset + 1_000_000, &device);
            offset += num_kv_heads * new_len * head_dim;

            // Simulate the real forward path: v arrives transposed (so
            // non-contiguous); k arrives contiguous. We build k and v as
            // contiguous here for simplicity, but also exercise a
            // non-contiguous v path by feeding a transposed layout every
            // other step.
            let v_input = if step % 2 == 0 {
                // Non-contiguous v: construct `[1, new_len, num_kv_heads, head_dim]`
                // and transpose(1, 2) to match the live forward shape.
                let (n1, n2, n3, n4) = (1, new_len, num_kv_heads, head_dim);
                let total = n1 * n2 * n3 * n4;
                let data: Vec<f32> = (0..total).map(|i| (offset + i) as f32 * 1e-3).collect();
                Tensor::from_vec(data, (n1, n2, n3, n4), &device).unwrap().transpose(1, 2).unwrap()
            } else {
                v_new.clone()
            };

            let (k_ss, v_ss) = cache_ss.append(&k_new, &v_input).unwrap();
            let (k_ip, v_ip) = cache_ip.append(&k_new, &v_input).unwrap();

            // Shape sanity.
            assert_eq!(k_ss.shape(), k_ip.shape(), "k shape mismatch at step {step}");
            assert_eq!(v_ss.shape(), v_ip.shape(), "v shape mismatch at step {step}");

            // Strict element-wise equality — the math is identical.
            //
            // `k_ip` / `v_ip` are narrow views (stride-!=-contiguous);
            // `k_ss` / `v_ss` are contiguous copies. `to_vec1` collapses
            // both through `to_dtype().flatten_all()`, which materializes
            // a standard row-major dense tensor via candle's copy path,
            // so the comparison walks each element through its strided
            // layout. Any stride gotcha in the in-place path will show
            // up immediately as a non-zero diff.
            let kd = max_abs_diff_strict(&k_ss, &k_ip);
            let vd = max_abs_diff_strict(&v_ss, &v_ip);
            eprintln!(
                "[kv-equiv] step {step} new_len={new_len} current_len={} \
                 k_diff={kd:.3e} v_diff={vd:.3e}",
                cache_ss.current_len,
            );
            k_diffs.push(kd);
            v_diffs.push(vd);

            // Critical: confirm the in-place narrow view is actually
            // non-contiguous in the decode-step case (visible_len <
            // cache_size), proving the stride-aware path is exercised.
            if cache_ip.current_len < cache_ip.cache_size {
                assert!(
                    !k_ip.is_contiguous() || cache_ip.current_len == cache_ip.cache_size,
                    "step {step}: in-place k view unexpectedly contiguous \
                     (current_len={} cache_size={})",
                    cache_ip.current_len,
                    cache_ip.cache_size,
                );
            }
        }

        (k_diffs, v_diffs)
    }

    /// Sliding layer (head_dim=256) — single-token decode sequence up to
    /// and past the sliding window boundary. This is the primary hot
    /// path for Gemma 4 sliding-attention layers during decode.
    #[test]
    fn test_kv_in_place_sliding_decode() {
        let num_kv_heads = 16;
        let head_dim = 256;
        let sliding_window = 1024_usize;

        // 1 prefill of 8, then 120 single-token appends. After 128 total
        // the cache is well inside the 1024-element sliding window.
        let mut lengths = vec![8_usize];
        lengths.extend(std::iter::repeat(1_usize).take(120));

        let (kd, vd) = run_kv_equivalence(num_kv_heads, head_dim, Some(sliding_window), &lengths);
        let max_kd = kd.iter().copied().fold(0.0_f64, f64::max);
        let max_vd = vd.iter().copied().fold(0.0_f64, f64::max);
        eprintln!("[kv-sliding-decode] max k diff = {max_kd:.3e}, max v diff = {max_vd:.3e}");
        assert_eq!(max_kd, 0.0, "sliding-decode k path bit-mismatch");
        assert_eq!(max_vd, 0.0, "sliding-decode v path bit-mismatch");
    }

    /// Global layer (head_dim=512) — single-token decode sequence.
    /// Covers the Gemma 4 global-attention full-history path.
    #[test]
    fn test_kv_in_place_global_decode() {
        let num_kv_heads = 16;
        let head_dim = 512;

        // 1 prefill of 8, then 120 single-token appends. Global layers
        // have sliding_window = None, so visible_len == current_len
        // throughout.
        let mut lengths = vec![8_usize];
        lengths.extend(std::iter::repeat(1_usize).take(120));

        let (kd, vd) = run_kv_equivalence(num_kv_heads, head_dim, None, &lengths);
        let max_kd = kd.iter().copied().fold(0.0_f64, f64::max);
        let max_vd = vd.iter().copied().fold(0.0_f64, f64::max);
        eprintln!("[kv-global-decode] max k diff = {max_kd:.3e}, max v diff = {max_vd:.3e}");
        assert_eq!(max_kd, 0.0, "global-decode k path bit-mismatch");
        assert_eq!(max_vd, 0.0, "global-decode v path bit-mismatch");
    }

    /// Prefill path: one big append that fills a meaningful portion of
    /// the cache. Exercises the consumer's expectation of `q_len > 1`.
    #[test]
    fn test_kv_in_place_prefill() {
        let num_kv_heads = 16;
        let head_dim = 256;
        let sliding_window = 1024_usize;

        // Single 187-token prefill (matches the canonical bench prompt length).
        let lengths = vec![187_usize];
        let (kd, vd) = run_kv_equivalence(num_kv_heads, head_dim, Some(sliding_window), &lengths);
        let max_kd = kd.iter().copied().fold(0.0_f64, f64::max);
        let max_vd = vd.iter().copied().fold(0.0_f64, f64::max);
        eprintln!("[kv-prefill] max k diff = {max_kd:.3e}, max v diff = {max_vd:.3e}");
        assert_eq!(max_kd, 0.0, "prefill k path bit-mismatch");
        assert_eq!(max_vd, 0.0, "prefill v path bit-mismatch");
    }

    /// Sliding-window truncation: append past the window boundary and
    /// verify the visible range (last `W` tokens) matches between modes.
    /// This covers the sliding-layer continuation hot path once the
    /// sequence exceeds `sliding_window`.
    #[test]
    fn test_kv_in_place_sliding_truncation() {
        let num_kv_heads = 8;
        let head_dim = 256;
        // Small window so the test runs fast and actually exercises the
        // `visible_start > 0` branch.
        let sliding_window = 32_usize;

        // 1 prefill of 16, then 40 single-token appends → current_len = 56,
        // which exceeds the 32-element window by 24. visible_len stays
        // pinned at 32 while `current_len` advances.
        let mut lengths = vec![16_usize];
        lengths.extend(std::iter::repeat(1_usize).take(40));

        let (kd, vd) = run_kv_equivalence(num_kv_heads, head_dim, Some(sliding_window), &lengths);
        let max_kd = kd.iter().copied().fold(0.0_f64, f64::max);
        let max_vd = vd.iter().copied().fold(0.0_f64, f64::max);
        eprintln!("[kv-sliding-trunc] max k diff = {max_kd:.3e}, max v diff = {max_vd:.3e}");
        assert_eq!(max_kd, 0.0, "sliding-truncation k path bit-mismatch");
        assert_eq!(max_vd, 0.0, "sliding-truncation v path bit-mismatch");
    }

    /// Stride-correctness spot check: verify the in-place decode view's
    /// strides match what the SDPA vector kernel expects
    /// (`k_stride[1] == cache_size * head_dim`). This is the check that
    /// guards against the class of bugs that caused the a0952e2
    /// regression.
    #[test]
    fn test_kv_in_place_decode_strides() {
        let device = Device::new_metal(0).expect("Metal device required");
        let num_kv_heads = 16_usize;
        let head_dim = 256_usize;

        let mut cache = KvCache::new(
            num_kv_heads,
            head_dim,
            &device,
            Some(1024),
            KvCacheKernelMode::InPlace,
        )
        .unwrap();
        // Start with a small prefill then drop in a decode token.
        let k = synthetic_kv(num_kv_heads, 8, head_dim, 0, &device);
        let v = synthetic_kv(num_kv_heads, 8, head_dim, 10_000, &device);
        let (_, _) = cache.append(&k, &v).unwrap();

        let k1 = synthetic_kv(num_kv_heads, 1, head_dim, 20_000, &device);
        let v1 = synthetic_kv(num_kv_heads, 1, head_dim, 30_000, &device);
        let (k_view, v_view) = cache.append(&k1, &v1).unwrap();

        let cache_size = cache.cache_size;
        let expected_dim1_stride = cache_size * head_dim;
        assert_eq!(
            k_view.stride()[1],
            expected_dim1_stride,
            "k_view dim-1 stride must be cache_size*head_dim={expected_dim1_stride} \
             (got {}); SDPA vector kernel reads this as `kstride`",
            k_view.stride()[1]
        );
        assert_eq!(
            v_view.stride()[1],
            expected_dim1_stride,
            "v_view dim-1 stride must be cache_size*head_dim={expected_dim1_stride} \
             (got {}); SDPA vector kernel reads this as `vstride`",
            v_view.stride()[1]
        );
        // Innermost (head_dim) stride must be 1 — SDPA vector kernels
        // read positions with assumed hd-packed layout.
        assert_eq!(k_view.stride()[3], 1, "k_view innermost stride must be 1");
        assert_eq!(v_view.stride()[3], 1, "v_view innermost stride must be 1");

        // Position stride must be head_dim — SDPA walks positions this way.
        assert_eq!(k_view.stride()[2], head_dim, "k_view position stride must be head_dim");
        assert_eq!(v_view.stride()[2], head_dim, "v_view position stride must be head_dim");

        eprintln!(
            "[kv-strides] cache_size={cache_size} k_view strides={:?} v_view strides={:?}",
            k_view.stride(),
            v_view.stride()
        );
    }

    /// ADR-005 1bNEW.20.FIX — SDPA must produce identical output when
    /// fed a non-contiguous K/V view with **non-zero** `start_offset`
    /// vs. the same data re-contiguified to zero offset.
    ///
    /// This test exists because the 1bNEW.20 Phase B gates (byte-identical
    /// 128-token decode, 827-token adversarial recall) all operated below
    /// the sliding-window boundary where `visible_start == 0`, so they
    /// never exercised a SDPA dispatch against a `start_offset > 0`
    /// view. A Gemma-4 sourdough-instruction prompt at `max_tokens=20000`
    /// flipped from coherent to garbage at exactly the sliding-window
    /// boundary (~decode token 1024), and the bisect landed on
    /// `candle_nn/src/ops.rs` passing `layout.start_offset()` directly
    /// into `candle_metal_kernels::call_sdpa_vector` et al. without the
    /// `* dtype.size_in_bytes()` multiplication that every other Metal
    /// op in candle applies. Metal interprets the value as a byte
    /// offset, so the kernel bound q/k/v at a garbage memory address
    /// whenever the narrow view had a non-zero element-count offset.
    ///
    /// The fix is vendored as `/opt/hf2q/vendor/candle-nn/src/ops.rs`
    /// (nine substitution sites across the three dispatch branches).
    /// This test would have failed on the stock candle-nn-0.10.2 and
    /// must continue to pass on the vendored patch. If the test starts
    /// failing, either the candle-nn patch was dropped or an upstream
    /// bump re-introduced the bug.
    ///
    /// The test is deliberately NOT coupled to `KvCache` — it
    /// constructs K/V directly via `narrow` on a pre-allocated
    /// contiguous tensor with `start > 0`. That isolates the bug to
    /// the exact dispatch-path condition that matters (SDPA input
    /// with `start_offset > 0`), without any KV-cache plumbing noise.
    #[test]
    fn test_candle_sdpa_honors_nonzero_start_offset() {
        use candle_nn::ops::sdpa;

        let device = Device::new_metal(0).expect("Metal device required");

        // Decode-path shapes: q_len=1, non-trivial K/V sequence, Gemma-4
        // sliding head_dim=256 (the exact path the sourdough bug hit).
        let num_q_heads = 16_usize;
        let num_kv_heads = 16_usize;
        let head_dim = 256_usize;
        let cache_size = 128_usize; // simulates the pre-allocated cache
        let visible_start = 37_usize; // simulates `current_len - sliding_window`
        let visible_len = 64_usize; // simulates `sliding_window`
        assert!(visible_start + visible_len <= cache_size);

        // Deterministic fill. Every element unique so a stride gotcha is
        // observable in the output, not masked by collisions.
        let ntotal = num_kv_heads * cache_size * head_dim;
        let k_full: Vec<f32> = (0..ntotal).map(|i| (i as f32) * 1e-4).collect();
        let v_full: Vec<f32> =
            (0..ntotal).map(|i| ((i + 1_000_000) as f32) * 1e-4).collect();
        let k_base =
            Tensor::from_vec(k_full, (1, num_kv_heads, cache_size, head_dim), &device).unwrap();
        let v_base =
            Tensor::from_vec(v_full, (1, num_kv_heads, cache_size, head_dim), &device).unwrap();

        // Stride-aware narrow with `visible_start > 0` — this is the
        // layout `KvCache::append_in_place` returns once a sliding
        // layer has advanced past its window.
        let k_view = k_base.narrow(2, visible_start, visible_len).unwrap();
        let v_view = v_base.narrow(2, visible_start, visible_len).unwrap();

        // Sanity: the narrow view is non-contiguous and carries a
        // non-zero layout `start_offset` in **elements**.
        assert!(!k_view.is_contiguous(), "k_view should not be contiguous");
        assert!(!v_view.is_contiguous(), "v_view should not be contiguous");

        // Control: re-materialize the same data with `start_offset == 0`.
        // The data is element-identical to `k_view` / `v_view`.
        let k_ref = k_view.contiguous().unwrap();
        let v_ref = v_view.contiguous().unwrap();
        assert!(k_ref.is_contiguous());
        assert!(v_ref.is_contiguous());

        // Query tensor — typical decode shape. Must be contiguous
        // (candle-nn SDPA contract).
        let q_total = num_q_heads * head_dim;
        let q_data: Vec<f32> = (0..q_total).map(|i| ((i + 2_000_000) as f32) * 1e-4).collect();
        let q = Tensor::from_vec(q_data, (1, num_q_heads, 1, head_dim), &device)
            .unwrap()
            .contiguous()
            .unwrap();

        let scale = 1.0_f32 / (head_dim as f32).sqrt();

        // Two SDPA calls on the SAME data, differing only in whether
        // the K/V arguments have `start_offset > 0`. If candle-nn's
        // dispatch converts element-offset to byte-offset correctly,
        // both calls produce bit-identical output.
        let out_view = sdpa(&q, &k_view, &v_view, None, false, scale, 1.0).unwrap();
        let out_ref = sdpa(&q, &k_ref, &v_ref, None, false, scale, 1.0).unwrap();

        let flat_view: Vec<f32> =
            out_view.to_dtype(DType::F32).unwrap().flatten_all().unwrap().to_vec1().unwrap();
        let flat_ref: Vec<f32> =
            out_ref.to_dtype(DType::F32).unwrap().flatten_all().unwrap().to_vec1().unwrap();
        assert_eq!(flat_view.len(), flat_ref.len());

        let max_diff = flat_view
            .iter()
            .zip(flat_ref.iter())
            .map(|(a, b)| (*a - *b).abs())
            .fold(0.0_f32, f32::max);
        let max_abs_ref = flat_ref.iter().copied().fold(0.0_f32, f32::max);

        eprintln!(
            "[sdpa-offset] visible_start={visible_start} visible_len={visible_len} \
             max_diff={max_diff:.6e} max_abs_ref={max_abs_ref:.6e}"
        );

        // Tight tolerance: SDPA vector kernels are deterministic at
        // equal inputs, so the only allowed divergence is floating
        // accumulation order within a simdgroup, which is identical
        // between the two calls because the tensor shape is identical
        // and the only difference is the base buffer offset. Expect
        // bit-identical (0.0), assert with a wide safety margin so
        // any real divergence screams.
        assert!(
            max_diff < 1e-5,
            "SDPA output diverges between stride-aware narrow view and \
             contiguous ref (max_diff={max_diff:.6e}). Likely candle-nn \
             SDPA dispatch regressed on element-offset → byte-offset fix. \
             See /opt/hf2q/vendor/candle-nn/src/ops.rs header."
        );
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn load_qlinear(
    gguf: &GgufModel,
    prefix: &str,
    counters: &Arc<DispatchCounters>,
) -> Result<QLinear> {
    let qt = gguf.get_qtensor(&format!("{}.weight", prefix))?;
    let qmm = QMatMul::from_arc(qt)?;
    let bias = gguf.try_get_tensor(&format!("{}.bias", prefix), MODEL_DTYPE)?;
    Ok(QLinear::new(qmm, bias, counters.clone()))
}

fn load_rms_norm(
    gguf: &GgufModel,
    prefix: &str,
    eps: f64,
    counters: &Arc<DispatchCounters>,
    kernel: &RmsNormKernel,
) -> Result<RmsNorm> {
    let weight = gguf.get_tensor(&format!("{}.weight", prefix), MODEL_DTYPE)?;
    Ok(RmsNorm::new(weight, eps, counters.clone(), kernel.clone()))
}

/// Map `candle_core::quantized::GgmlDType` to the sister
/// `candle_metal_kernels::GgmlDType` tag used by
/// `moe_kernel::call_quantized_matmul_mv_id_t`. The two enums are
/// logically identical but live in separate crates that do not have a
/// `From` impl, so we do the mapping explicitly here.
///
/// Only the quant types actually shipped by the Gemma 4 A4B MoE experts
/// (`Q6K` for `ffn_gate_up_exps`, `Q8_0` for `ffn_down_exps`) are
/// supported in the fused path. Adding a new type is a one-line change
/// under Walk discipline.
#[cfg(feature = "metal")]
fn core_to_metal_ggml(
    t: candle_core::quantized::GgmlDType,
) -> Result<MetalGgmlDType> {
    use candle_core::quantized::GgmlDType as CoreT;
    Ok(match t {
        CoreT::Q4_0 => MetalGgmlDType::Q4_0,
        CoreT::Q4_1 => MetalGgmlDType::Q4_1,
        CoreT::Q5_0 => MetalGgmlDType::Q5_0,
        CoreT::Q5_1 => MetalGgmlDType::Q5_1,
        CoreT::Q8_0 => MetalGgmlDType::Q8_0,
        CoreT::Q2K => MetalGgmlDType::Q2K,
        CoreT::Q3K => MetalGgmlDType::Q3K,
        CoreT::Q4K => MetalGgmlDType::Q4K,
        CoreT::Q5K => MetalGgmlDType::Q5K,
        CoreT::Q6K => MetalGgmlDType::Q6K,
        other => anyhow::bail!(
            "core_to_metal_ggml: dtype {other:?} not supported by fused MoE dispatch. \
             Gemma 4 experts ship as Q6K (gate_up) + Q8_0 (down)."
        ),
    })
}

// ---------------------------------------------------------------------------
// Debug Forward-Pass Component Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod forward_tests {
    use super::*;
    use candle_core::{DType, Device, Tensor};
    use candle_core::quantized::QMatMul;
    use std::path::Path;

    const GGUF_PATH: &str =
        "models/gemma-4-26B-A4B-it-ara-abliterated-dwq/gemma-4-26B-A4B-it-ara-abliterated-dwq.gguf";
    const CONFIG_PATH: &str = "models/gemma4/config.json";

    /// Helper: print first N f32 values from a tensor (flattened).
    fn first_n(t: &Tensor, n: usize) -> Vec<f32> {
        t.to_dtype(DType::F32)
            .and_then(|t| t.flatten_all())
            .and_then(|t| t.to_vec1::<f32>())
            .map(|v| v.into_iter().take(n).collect())
            .unwrap_or_default()
    }

    /// Helper: max absolute difference between two tensors.
    fn max_abs_diff(a: &Tensor, b: &Tensor) -> f32 {
        let a_flat: Vec<f32> = a.to_dtype(DType::F32).unwrap().flatten_all().unwrap().to_vec1().unwrap();
        let b_flat: Vec<f32> = b.to_dtype(DType::F32).unwrap().flatten_all().unwrap().to_vec1().unwrap();
        a_flat.iter().zip(b_flat.iter())
            .map(|(x, y)| (x - y).abs())
            .fold(0.0f32, f32::max)
    }

    fn load_test_fixtures() -> (GgufModel, Gemma4Config, Device) {
        let device = Device::new_metal(0).expect("Metal device required for these tests");
        let gguf = GgufModel::load(Path::new(GGUF_PATH), &device)
            .expect("Failed to load GGUF model");
        let cfg = Gemma4Config::from_config_json(Path::new(CONFIG_PATH))
            .expect("Failed to load config");
        (gguf, cfg, device)
    }

    /// Per-test throwaway counters — none of these tests care about dispatch
    /// counting, but the constructors now require a counter Arc.
    fn test_counters() -> Arc<DispatchCounters> {
        DispatchCounters::new()
    }

    /// Per-test throwaway RmsNorm kernel bundle. All of these tests
    /// are `#[ignore]` and exercise the `loop` path for compile
    /// coverage only; the fused kernel has its own unit tests in
    /// `rms_norm_kernel::tests`.
    fn test_rms_kernel() -> RmsNormKernel {
        RmsNormKernel::loop_mode()
    }

    // -----------------------------------------------------------------------
    // Test 1: Embedding lookup + scale
    // -----------------------------------------------------------------------

    #[test]
    #[ignore]
    fn test_forward_01_embedding() {
        let (gguf, cfg, _device) = load_test_fixtures();

        let embed_w = gguf.get_tensor("token_embd.weight", MODEL_DTYPE).unwrap();
        // Transpose if GGUF stored as [hidden, vocab]
        let embed_w = if embed_w.dim(0).unwrap() == cfg.hidden_size
            && embed_w.dim(1).unwrap() == cfg.vocab_size
        {
            embed_w.t().unwrap().contiguous().unwrap()
        } else {
            embed_w
        };

        println!("\n=== Test 1: Embedding ===");
        println!("token_embd.weight shape: {:?}", embed_w.shape());

        // Look up token ID 2 (BOS)
        let embed = candle_nn::Embedding::new(embed_w, cfg.hidden_size);
        let token_ids = Tensor::new(&[2u32], embed.embeddings().device()).unwrap();
        let token_ids = token_ids.unsqueeze(0).unwrap(); // [1, 1]
        let emb = embed.forward(&token_ids).unwrap(); // [1, 1, hidden]

        let scale = (cfg.hidden_size as f64).sqrt();
        let scaled = (&emb * scale).unwrap();

        println!("BOS embedding (raw) first 5: {:?}", first_n(&emb, 5));
        println!("Scale factor: {}", scale);
        println!("BOS embedding (scaled) first 5: {:?}", first_n(&scaled, 5));
        println!("Embedding dtype: {:?}", scaled.dtype());
    }

    // -----------------------------------------------------------------------
    // Test 2: QMatMul vs dequantized matmul
    // -----------------------------------------------------------------------

    #[test]
    #[ignore]
    fn test_forward_02_qmatmul_vs_dequant() {
        let (gguf, _cfg, device) = load_test_fixtures();

        println!("\n=== Test 2: QMatMul vs Dequantize ===");

        // Load as QMatMul
        let qt = gguf.get_qtensor("blk.0.attn_q.weight").unwrap();
        let qmm = QMatMul::from_arc(qt).unwrap();

        // Load as dequantized tensor
        let deq_w = gguf.get_tensor("blk.0.attn_q.weight", DType::F32).unwrap();
        println!("Dequantized weight shape: {:?}", deq_w.shape());

        // Input: ones(1, 2816)
        let input = Tensor::ones((1, 2816), DType::F32, &device).unwrap();

        // QMatMul forward (does x @ W^T internally)
        let qmm_out = qmm.forward(&input).unwrap();
        println!("QMatMul output shape: {:?}", qmm_out.shape());
        println!("QMatMul first 5: {:?}", first_n(&qmm_out, 5));

        // Manual: input @ deq_w.T
        let manual_out = input.matmul(&deq_w.t().unwrap()).unwrap();
        println!("Dequant+matmul output shape: {:?}", manual_out.shape());
        println!("Dequant+matmul first 5: {:?}", first_n(&manual_out, 5));

        let diff = max_abs_diff(&qmm_out, &manual_out);
        println!("Max abs diff: {}", diff);
        assert!(diff < 1.0, "QMatMul vs dequant+matmul differ by more than 1.0: {}", diff);
    }

    // -----------------------------------------------------------------------
    // Test 3: RmsNorm
    // -----------------------------------------------------------------------

    #[test]
    #[ignore]
    fn test_forward_03_rmsnorm() {
        let (gguf, cfg, device) = load_test_fixtures();

        println!("\n=== Test 3: RmsNorm ===");

        let norm_w = gguf.get_tensor("blk.0.attn_norm.weight", MODEL_DTYPE).unwrap();
        println!("attn_norm.weight shape: {:?}", norm_w.shape());
        println!("attn_norm.weight first 5: {:?}", first_n(&norm_w, 5));

        let norm = RmsNorm::new(norm_w, cfg.rms_norm_eps, test_counters(), test_rms_kernel());

        // Input: ones(1, 2816)
        let input = Tensor::ones((1, 2816), MODEL_DTYPE, &device).unwrap();
        let output = norm.forward(&input).unwrap();
        println!("RmsNorm(ones) first 5: {:?}", first_n(&output, 5));
        println!("  (should equal the norm weight values since ones are normalized to 1.0)");

        // Input: arange for non-trivial test
        let arange = Tensor::arange(0u32, 2816, &device).unwrap()
            .to_dtype(DType::F32).unwrap()
            .unsqueeze(0).unwrap()
            .to_dtype(MODEL_DTYPE).unwrap();
        let output2 = norm.forward(&arange).unwrap();
        println!("RmsNorm(arange) first 5: {:?}", first_n(&output2, 5));
    }

    // -----------------------------------------------------------------------
    // Test 4: Single attention layer Q/K/V projections
    // -----------------------------------------------------------------------

    #[test]
    #[ignore]
    fn test_forward_04_attention_projections() {
        let (gguf, cfg, device) = load_test_fixtures();

        println!("\n=== Test 4: Attention Projections (layer 0) ===");

        // Layer 0 is sliding attention
        let is_full = cfg.is_full_attention(0);
        let head_dim = cfg.head_dim_for_layer(0);
        let num_kv_heads = cfg.num_kv_heads_for_layer(0);
        println!("Layer 0: is_full={}, head_dim={}, num_heads={}, num_kv_heads={}",
            is_full, head_dim, cfg.num_attention_heads, num_kv_heads);

        // Load Q, K, V projections
        let ctrs = test_counters();
        let q_proj = load_qlinear(&gguf, "blk.0.attn_q", &ctrs).unwrap();
        let k_proj = load_qlinear(&gguf, "blk.0.attn_k", &ctrs).unwrap();
        let v_proj = load_qlinear(&gguf, "blk.0.attn_v", &ctrs).unwrap();

        // Get the BOS embedding as input
        let embed_w = gguf.get_tensor("token_embd.weight", MODEL_DTYPE).unwrap();
        let embed_w = if embed_w.dim(0).unwrap() == cfg.hidden_size
            && embed_w.dim(1).unwrap() == cfg.vocab_size
        {
            embed_w.t().unwrap().contiguous().unwrap()
        } else {
            embed_w
        };
        let embed = candle_nn::Embedding::new(embed_w, cfg.hidden_size);
        let token_ids = Tensor::new(&[2u32], &device).unwrap().unsqueeze(0).unwrap();
        let emb = embed.forward(&token_ids).unwrap();
        let scale = (cfg.hidden_size as f64).sqrt();
        let input = (&emb * scale).unwrap(); // [1, 1, hidden]
        println!("Input (scaled BOS) shape: {:?}, first 5: {:?}", input.shape(), first_n(&input, 5));

        // Q projection
        let q_out = q_proj.forward(&input.squeeze(0).unwrap()).unwrap();
        let expected_q_dim = cfg.num_attention_heads * head_dim;
        println!("Q projection output shape: {:?} (expected [1, {}])", q_out.shape(), expected_q_dim);
        println!("Q first 5: {:?}", first_n(&q_out, 5));

        // K projection
        let k_out = k_proj.forward(&input.squeeze(0).unwrap()).unwrap();
        let expected_k_dim = num_kv_heads * head_dim;
        println!("K projection output shape: {:?} (expected [1, {}])", k_out.shape(), expected_k_dim);
        println!("K first 5: {:?}", first_n(&k_out, 5));

        // V projection
        let v_out = v_proj.forward(&input.squeeze(0).unwrap()).unwrap();
        println!("V projection output shape: {:?} (expected [1, {}])", v_out.shape(), expected_k_dim);
        println!("V first 5: {:?}", first_n(&v_out, 5));

        // Q/K norms
        let rk = test_rms_kernel();
        let q_norm = load_rms_norm(&gguf, "blk.0.attn_q_norm", cfg.rms_norm_eps, &ctrs, &rk).unwrap();
        let k_norm = load_rms_norm(&gguf, "blk.0.attn_k_norm", cfg.rms_norm_eps, &ctrs, &rk).unwrap();

        // Reshape Q to [1, 1, num_heads, head_dim] for norm
        let q_reshaped = q_out.reshape((1, 1, cfg.num_attention_heads, head_dim)).unwrap();
        let q_normed = q_norm.forward(&q_reshaped).unwrap();
        println!("Q after norm, first 5: {:?}", first_n(&q_normed, 5));

        let k_reshaped = k_out.reshape((1, 1, num_kv_heads, head_dim)).unwrap();
        let k_normed = k_norm.forward(&k_reshaped).unwrap();
        println!("K after norm, first 5: {:?}", first_n(&k_normed, 5));
    }

    // -----------------------------------------------------------------------
    // Test 5: MoE router
    // -----------------------------------------------------------------------

    #[test]
    #[ignore]
    fn test_forward_05_moe_router() {
        let (gguf, cfg, device) = load_test_fixtures();

        println!("\n=== Test 5: MoE Router (layer 0) ===");

        // Load router weights
        let ctrs = test_counters();
        let router_proj = load_qlinear(&gguf, "blk.0.ffn_gate_inp", &ctrs).unwrap();
        let router_scale = gguf.get_tensor("blk.0.ffn_gate_inp.scale", MODEL_DTYPE).unwrap();
        println!("Router scale shape: {:?}, first 5: {:?}", router_scale.shape(), first_n(&router_scale, 5));

        let per_expert_scale = gguf.get_tensor("blk.0.ffn_down_exps.scale", MODEL_DTYPE).unwrap();
        println!("Per-expert scale shape: {:?}, first 5: {:?}", per_expert_scale.shape(), first_n(&per_expert_scale, 5));

        // Create a simple hidden state: ones(1, hidden)
        let input = Tensor::ones((1, cfg.hidden_size), MODEL_DTYPE, &device).unwrap();

        // Router pipeline: unit_rms_norm → scale → (1/sqrt(hidden)) → project → softmax
        let normed = rms_norm_unit(&input, &ctrs, &test_rms_kernel()).unwrap();
        println!("After unit rms_norm first 5: {:?}", first_n(&normed, 5));

        let scale_factor = (cfg.hidden_size as f64).powf(-0.5);
        println!("Scale factor (1/sqrt({})): {}", cfg.hidden_size, scale_factor);

        let scaled = (normed.broadcast_mul(&router_scale).unwrap() * scale_factor).unwrap();
        println!("After scale first 5: {:?}", first_n(&scaled, 5));

        let logits = router_proj.forward(&scaled).unwrap();
        println!("Router logits shape: {:?}", logits.shape());
        println!("Router logits first 5: {:?}", first_n(&logits, 5));

        let probs = candle_nn::ops::softmax_last_dim(&logits).unwrap();
        println!("Router probs first 5: {:?}", first_n(&probs, 5));

        // Find top-8 experts
        let probs_vec: Vec<f32> = probs.to_dtype(DType::F32).unwrap().flatten_all().unwrap().to_vec1().unwrap();
        let per_expert_vec: Vec<f32> = per_expert_scale.to_dtype(DType::F32).unwrap().flatten_all().unwrap().to_vec1().unwrap();

        let mut indexed: Vec<(usize, f32)> = probs_vec.iter().copied().enumerate().collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        println!("Top-8 experts:");
        let sum: f32 = indexed.iter().take(cfg.top_k_experts).map(|(_, w)| w).sum();
        for &(idx, weight) in indexed.iter().take(cfg.top_k_experts) {
            let normalized = weight / sum;
            let expert_scale = if idx < per_expert_vec.len() { per_expert_vec[idx] } else { 1.0 };
            println!("  Expert {}: raw_prob={:.6}, normalized={:.6}, per_expert_scale={:.6}",
                idx, weight, normalized, expert_scale);
        }
        println!("Sum of top-{} probs: {:.6}", cfg.top_k_experts, sum);
    }

    // -----------------------------------------------------------------------
    // Test 6: Expert forward (expert 0, layer 0)
    // -----------------------------------------------------------------------

    #[test]
    #[ignore]
    fn test_forward_06_expert() {
        let (gguf, cfg, device) = load_test_fixtures();

        println!("\n=== Test 6: Expert Forward (expert 0, layer 0) ===");

        // Load 3D expert weights
        let gate_up_3d = gguf.get_tensor("blk.0.ffn_gate_up_exps.weight", MODEL_DTYPE).unwrap();
        let down_3d = gguf.get_tensor("blk.0.ffn_down_exps.weight", MODEL_DTYPE).unwrap();

        println!("gate_up_exps raw shape (candle): {:?}", gate_up_3d.shape());
        println!("down_exps raw shape (candle): {:?}", down_3d.shape());

        // Slice expert 0, transpose to [in, out] for matmul
        let gate_up_e0 = gate_up_3d.narrow(0, 0, 1).unwrap().squeeze(0).unwrap();
        println!("Expert 0 gate_up slice shape: {:?}", gate_up_e0.shape());
        let gate_up_w = gate_up_e0.t().unwrap().contiguous().unwrap();
        println!("Expert 0 gate_up (transposed) shape: {:?}", gate_up_w.shape());

        let down_e0 = down_3d.narrow(0, 0, 1).unwrap().squeeze(0).unwrap();
        println!("Expert 0 down slice shape: {:?}", down_e0.shape());
        let down_w = down_e0.t().unwrap().contiguous().unwrap();
        println!("Expert 0 down (transposed) shape: {:?}", down_w.shape());

        // Input: ones(1, hidden)
        let input = Tensor::ones((1, cfg.hidden_size), MODEL_DTYPE, &device).unwrap();

        // gate_up = input @ gate_up_w
        let gate_up = input.matmul(&gate_up_w).unwrap();
        println!("gate_up output shape: {:?} (expected [1, {}])",
            gate_up.shape(), cfg.moe_intermediate_size * 2);
        println!("gate_up first 5: {:?}", first_n(&gate_up, 5));

        // Split into gate and up
        let gate = gate_up.narrow(1, 0, cfg.moe_intermediate_size).unwrap();
        let up = gate_up.narrow(1, cfg.moe_intermediate_size, cfg.moe_intermediate_size).unwrap();
        println!("gate shape: {:?}, first 5: {:?}", gate.shape(), first_n(&gate, 5));
        println!("up shape: {:?}, first 5: {:?}", up.shape(), first_n(&up, 5));

        // Apply GELU to gate, multiply by up
        let gate_act = candle_nn::Activation::GeluPytorchTanh.forward(&gate).unwrap();
        println!("gate after GELU first 5: {:?}", first_n(&gate_act, 5));

        let fused = (&gate_act * &up).unwrap();
        println!("fused (gate*up) shape: {:?}, first 5: {:?}", fused.shape(), first_n(&fused, 5));

        // down = fused @ down_w
        let down_out = fused.matmul(&down_w).unwrap();
        println!("down output shape: {:?} (expected [1, {}])", down_out.shape(), cfg.hidden_size);
        println!("down first 5: {:?}", first_n(&down_out, 5));

        // Check for NaN/Inf
        let flat: Vec<f32> = down_out.to_dtype(DType::F32).unwrap().flatten_all().unwrap().to_vec1().unwrap();
        let nan_count = flat.iter().filter(|v| v.is_nan()).count();
        let inf_count = flat.iter().filter(|v| v.is_infinite()).count();
        println!("NaN count: {}, Inf count: {}", nan_count, inf_count);
    }

    // -----------------------------------------------------------------------
    // Test 7: Full layer 0 forward pass (dense MLP branch only)
    // -----------------------------------------------------------------------

    #[test]
    #[ignore]
    fn test_forward_07_dense_mlp() {
        let (gguf, cfg, device) = load_test_fixtures();

        println!("\n=== Test 7: Dense MLP (layer 0) ===");

        // Load MLP weights
        let ctrs = test_counters();
        let gate_proj = load_qlinear(&gguf, "blk.0.ffn_gate", &ctrs).unwrap();
        let up_proj = load_qlinear(&gguf, "blk.0.ffn_up", &ctrs).unwrap();
        let down_proj = load_qlinear(&gguf, "blk.0.ffn_down", &ctrs).unwrap();

        let mlp = Mlp { gate_proj, up_proj, down_proj, counters: ctrs };

        // Input: ones(1, hidden)
        let input = Tensor::ones((1, cfg.hidden_size), MODEL_DTYPE, &device).unwrap();

        let gate_out = mlp.gate_proj.forward(&input).unwrap();
        println!("gate_proj output shape: {:?}, first 5: {:?}", gate_out.shape(), first_n(&gate_out, 5));

        let up_out = mlp.up_proj.forward(&input).unwrap();
        println!("up_proj output shape: {:?}, first 5: {:?}", up_out.shape(), first_n(&up_out, 5));

        let mlp_out = mlp.forward(&input).unwrap();
        println!("Dense MLP output shape: {:?}, first 5: {:?}", mlp_out.shape(), first_n(&mlp_out, 5));

        // NaN/Inf check
        let flat: Vec<f32> = mlp_out.to_dtype(DType::F32).unwrap().flatten_all().unwrap().to_vec1().unwrap();
        let nan_count = flat.iter().filter(|v| v.is_nan()).count();
        let inf_count = flat.iter().filter(|v| v.is_infinite()).count();
        println!("NaN count: {}, Inf count: {}", nan_count, inf_count);
    }

    // -----------------------------------------------------------------------
    // Test 8: Layer scalar + norms pipeline
    // -----------------------------------------------------------------------

    #[test]
    #[ignore]
    fn test_forward_08_layer_scalar_and_norms() {
        let (gguf, cfg, device) = load_test_fixtures();

        println!("\n=== Test 8: Layer Scalar and Post-Norms (layer 0) ===");

        let layer_scalar = gguf.get_tensor("blk.0.layer_output_scale.weight", MODEL_DTYPE).unwrap();
        println!("layer_scalar shape: {:?}, first 5: {:?}", layer_scalar.shape(), first_n(&layer_scalar, 5));

        let ctrs = test_counters();
        let rk = test_rms_kernel();
        let post_attn_norm = load_rms_norm(&gguf, "blk.0.post_attention_norm", cfg.rms_norm_eps, &ctrs, &rk).unwrap();
        let post_ffw_norm = load_rms_norm(&gguf, "blk.0.post_ffw_norm", cfg.rms_norm_eps, &ctrs, &rk).unwrap();
        let post_ffw_norm_1 = load_rms_norm(&gguf, "blk.0.post_ffw_norm_1", cfg.rms_norm_eps, &ctrs, &rk).unwrap();
        let post_ffw_norm_2 = load_rms_norm(&gguf, "blk.0.post_ffw_norm_2", cfg.rms_norm_eps, &ctrs, &rk).unwrap();

        // Test post-attention norm with ones
        let input = Tensor::ones((1, cfg.hidden_size), MODEL_DTYPE, &device).unwrap();
        let normed = post_attn_norm.forward(&input).unwrap();
        println!("post_attention_norm(ones) first 5: {:?}", first_n(&normed, 5));

        // Test layer scalar multiplication
        let scaled = input.broadcast_mul(&layer_scalar).unwrap();
        println!("ones * layer_scalar first 5: {:?}", first_n(&scaled, 5));

        // Print all post-FFW norm weights for comparison
        println!("post_ffw_norm weight first 5: {:?}", first_n(&post_ffw_norm.weight, 5));
        println!("post_ffw_norm_1 weight first 5: {:?}", first_n(&post_ffw_norm_1.weight, 5));
        println!("post_ffw_norm_2 weight first 5: {:?}", first_n(&post_ffw_norm_2.weight, 5));
    }

    // -----------------------------------------------------------------------
    // Test 9: End-to-end single token (BOS) through full model
    // -----------------------------------------------------------------------

    #[test]
    #[ignore]
    fn test_forward_09_single_token_e2e() {
        let (gguf, cfg, device) = load_test_fixtures();

        println!("\n=== Test 9: Single Token (BOS) End-to-End ===");

        let mut model = Gemma4Model::load(&cfg, &gguf, &device)
            .expect("Failed to load full model");

        let input_ids = Tensor::new(&[2u32], &device).unwrap().unsqueeze(0).unwrap();
        println!("Input token IDs: [2] (BOS)");

        let logits = model.forward(&input_ids, 0).unwrap();
        println!("Logits shape: {:?}", logits.shape());

        let logits_flat: Vec<f32> = logits.to_dtype(DType::F32).unwrap()
            .flatten_all().unwrap().to_vec1().unwrap();

        println!("Logits first 5: {:?}", &logits_flat[..5.min(logits_flat.len())]);

        // Find top-5 token predictions
        let mut indexed: Vec<(usize, f32)> = logits_flat.iter().copied().enumerate().collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        println!("Top-5 predicted tokens:");
        for &(idx, logit) in indexed.iter().take(5) {
            println!("  token {}: logit {:.4}", idx, logit);
        }

        // NaN/Inf check
        let nan_count = logits_flat.iter().filter(|v| v.is_nan()).count();
        let inf_count = logits_flat.iter().filter(|v| v.is_infinite()).count();
        println!("NaN count: {}, Inf count: {} (out of {})", nan_count, inf_count, logits_flat.len());
    }
}
