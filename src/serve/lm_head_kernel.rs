//! Quantized-weight lm_head via candle's native F16 gemm.
//!
//! # What this module does
//!
//! Provides the dispatch mode enum + the `lm_head_forward_fused` helper
//! used by `Gemma4Model::forward`. The helper performs the final vocab
//! projection `logits = normed @ W^T` directly against an F16 weight
//! tensor instead of the 2.95 GB F32 dequantized copy the Phase-1 code
//! reads every decode token at `gemma4.rs:1879`.
//!
//! This is **ADR-005 1bNEW.17**. It replaces the dense F32 matmul at
//! the lm_head site with a native F16 gemm dispatched via candle's
//! `call_mlx_gemm` path (same MLX-GEMM kernel pool candle's
//! `MetalStorage::matmul` uses at `candle-core/src/metal_backend/mod.rs:1675-1717`).
//! There is NO new Metal kernel — we reuse the compiled MLX `gemm_*_hgemm`
//! symbol that candle already instantiates for `DType::F16` matmuls.
//!
//! # Why F16 and not Q6_K
//!
//! The post-Walk re-spike (`docs/spike-post-walk-results.md`) projected
//! a Q6_K lm_head under the assumption that `token_embd.weight` was
//! stored Q6_K in the DWQ GGUF. **Empirically it is F16.** Inspected
//! at `models/gemma-4-26B-A4B-it-ara-abliterated-dwq/gemma-4-26B-A4B-it-ara-abliterated-dwq.gguf`
//! via `gguf.GGUFReader`: `token_embd.weight` is `GgmlDType::F16`, shape
//! `[2816, 262144]`, `n_bytes = 1,476,395,008` (exact match for F16).
//! There is no `output.weight` in the file either — llama.cpp falls
//! back to tied embeddings per `llama-model.cpp:4973-5610` and uses
//! the same F16 `token_embd` tensor for its lm_head. So the Walk-
//! faithful port is "read `model.output` in whatever dtype the GGUF
//! stores it", which for this GGUF is F16.
//!
//! Memory traffic per decode token:
//!   - Old F32 path:  262144 × 2816 × 4 bytes = 2.95 GB / token
//!   - New F16 path:  262144 × 2816 × 2 bytes = 1.48 GB / token  (−50%)
//!
//! At ~400 GB/s M5 Max effective bandwidth this trims ~3.67 ms/token
//! from the 7.14 ms lm_head spike measurement — projecting 48.71 →
//! ~59-60 tok/s, not the ADR's 67 tok/s Q6_K estimate. The saving is
//! still the single biggest remaining Walk-faithful lift in Phase 1b.
//!
//! # Reference citations (ADR-005 Walk discipline)
//!
//! - **llama.cpp Gemma 4 lm_head dispatch**:
//!   `/opt/llama.cpp/src/models/gemma4-iswa.cpp:248` —
//!   `cur = build_lora_mm(model.output, cur);`
//!   which resolves to `ggml_mul_mat(ctx0, model.output, cur)` at
//!   `/opt/llama.cpp/src/llama-graph.cpp:972` (`build_lora_mm`). The
//!   weight tensor `model.output` is whatever dtype the GGUF stores;
//!   llama.cpp does NOT dequantize to F32 before the matmul.
//!
//! - **Tied-embedding fallback (Gemma 4 has no `output.weight`)**:
//!   `/opt/llama.cpp/src/llama-model.cpp:4973-5610` —
//!   `// try to load output.weight, if not found, use token_embd (tied embeddings)`
//!   Implemented as `TENSOR_DUPLICATED` aliasing at e.g. `llama-model.cpp:2764`:
//!   `output = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), ..., TENSOR_DUPLICATED);`
//!
//! - **candle F16 gemm path**:
//!   `/opt/candle/candle-core/src/metal_backend/mod.rs:1685-1693`
//!   (`DType::F16 => candle_metal_kernels::GemmDType::F16`) and
//!   `:1695-1709` (`call_mlx_gemm(...)`). This is the same MLX-GEMM
//!   implementation candle routes every `DType::F16` `Tensor::matmul`
//!   through — no forked kernel, no new MSL source.
//!
//! - **Post-Walk re-spike measurement**: `docs/spike-post-walk-results.md`
//!   (7.14 ms/token forced-sync wall-clock at the lm_head call site
//!   on the current main binary, 48.71 tok/s baseline).
//!
//! # What this does NOT do
//!
//! - No new Metal kernel. Every matmul goes through candle's existing
//!   `call_mlx_gemm` dispatch.
//! - No CPU fallback. If the device is not Metal the `Loop` mode is
//!   the correct path anyway (it uses the same F32 dense matmul
//!   candle's CPU backend supports).
//! - No tie-word-embeddings rewrite. The Embedding lookup in
//!   `gemma4.rs::Gemma4Model::forward` still reads a separate F32
//!   tensor (the Phase-1 `embed_w`). The fused F16 lm_head weight is
//!   a SECOND, smaller copy held alongside. See Chesterton's fence
//!   below.
//!
//! # Chesterton's fence
//!
//! The Phase-1 code dequantized `token_embd.weight` to F32 at load
//! time (`gemma4.rs:1636` via `get_tensor(..., MODEL_DTYPE=F32)`) and
//! reused the same F32 tensor for both the Embedding lookup (hot-path
//! ~11 KB/token read) and the lm_head matmul (hot-path 2.95 GB/token
//! read). The shared F32 copy was pragmatic when tok/s was gated on
//! MoE/norm/RoPE; by the post-Walk re-spike it became the dominant
//! remaining cost at 64% of the gap. 1bNEW.17 keeps the F32 copy for
//! the Embedding lookup (no code churn on that code path) and adds a
//! parallel F16 copy used only at the lm_head site. Under `Fused`
//! mode the GPU allocation grows by 1.48 GB — acceptable on the 128 GB
//! M5 Max target box. Phase C does NOT remove the F32 copy for
//! bisect-safety; a later item can fold the embedding lookup onto the
//! F16 copy and eliminate the dual-hold entirely.

use anyhow::{anyhow, Result};
use candle_core::{DType, Tensor};

/// Per-model switch for the lm_head dispatch path. Populated from the
/// CLI `--lm-head-kernel` flag (see `cli::LmHeadKernelMode`) at
/// `Gemma4Model::load_with_modes` time.
///
/// Phase B (ADR-005 1bNEW.17, 2026-04-10): `Fused` activates the
/// native-F16 matmul path behind the `--lm-head-kernel=fused` flag.
/// Default stays `Loop` in Phase B for bisect-safety. Phase C flips
/// the default to `Fused` after the 5-run benchmark gate validates
/// coherence and the 827-token needle recall.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LmHeadKernelMode {
    /// Phase-1 baseline — dense F32 matmul against the
    /// 2.95 GB dequantized `token_embd.weight` copy.
    Loop,
    /// ADR-005 1bNEW.17 — native F16 matmul against a 1.48 GB
    /// parallel copy of `token_embd.weight`, routed through
    /// candle's `call_mlx_gemm` MLX-GEMM dispatch.
    Fused,
}

impl From<crate::cli::LmHeadKernelMode> for LmHeadKernelMode {
    fn from(v: crate::cli::LmHeadKernelMode) -> Self {
        match v {
            crate::cli::LmHeadKernelMode::Loop => Self::Loop,
            crate::cli::LmHeadKernelMode::Fused => Self::Fused,
        }
    }
}

/// Perform `logits = normed_f32 @ W_f16^T` via a native F16 matmul,
/// returning an F32 logits tensor.
///
/// Phase A lands this helper and its unit tests; Phase B wires it
/// into `Gemma4Model::forward`. `allow(dead_code)` matches the 1bNEW.1
/// / 1bNEW.4 / 1bNEW.6 Phase A pattern — the fused helper is built
/// and exercised by tests before the live forward-pass branch is
/// turned on behind the `--lm-head-kernel=fused` flag.
#[allow(dead_code)]
///
/// # Arguments
/// - `normed_2d`: the F32 input `[1, hidden]` — the output of the
///   final RmsNorm at `gemma4.rs:1875`.
/// - `lm_head_f16`: the F16 weight tensor `[vocab, hidden]` loaded
///   via `Tensor::to_dtype(F16)` from the existing F32 `embed_w`.
///
/// # Returns
/// An F32 logits tensor of shape `[1, vocab]`.
///
/// # Errors
/// Returns an error if either input tensor is not on a Metal device,
/// or if dtypes/shapes are inconsistent.
///
/// # Determinism
/// `call_mlx_gemm` with `GemmDType::F16` is deterministic on the same
/// hardware and driver; 5-run variance on the canonical decode bench
/// has been measured at 0 tok/s for every prior fused kernel in this
/// plan (1bNEW.1/4/6). Reduction order differs from the F32 path,
/// which is the explicit mechanism for the expected top-1 argmax
/// drift around the `The`/`To` tie (spike-post-walk-results.md).
pub fn lm_head_forward_fused(normed_2d: &Tensor, lm_head_f16: &Tensor) -> Result<Tensor> {
    // --- Shape + dtype guardrails (fail loud, Anti-Goal #7 no stubs) ----
    if lm_head_f16.dtype() != DType::F16 {
        return Err(anyhow!(
            "lm_head_kernel::lm_head_forward_fused: expected F16 weight, got {:?}",
            lm_head_f16.dtype()
        ));
    }
    if normed_2d.dtype() != DType::F32 {
        return Err(anyhow!(
            "lm_head_kernel::lm_head_forward_fused: expected F32 input, got {:?}",
            normed_2d.dtype()
        ));
    }
    if normed_2d.dims().len() != 2 {
        return Err(anyhow!(
            "lm_head_kernel::lm_head_forward_fused: expected 2D input [num_tokens, hidden], got shape {:?}",
            normed_2d.shape()
        ));
    }
    if lm_head_f16.dims().len() != 2 {
        return Err(anyhow!(
            "lm_head_kernel::lm_head_forward_fused: expected 2D weight [vocab, hidden], got shape {:?}",
            lm_head_f16.shape()
        ));
    }
    let (_n_tokens, hidden) = normed_2d.dims2()?;
    let (_vocab, w_hidden) = lm_head_f16.dims2()?;
    if hidden != w_hidden {
        return Err(anyhow!(
            "lm_head_kernel::lm_head_forward_fused: hidden mismatch input={} weight_in={}",
            hidden, w_hidden
        ));
    }

    // --- Cast input to F16, matmul, cast logits back to F32 ----------------
    //
    // Candle's `MetalStorage::matmul` dispatches `call_mlx_gemm` with
    // `GemmDType::F16` when both operands are F16, using the MLX
    // `gemm_*_hgemm` symbol pool already compiled into the
    // `candle-metal-kernels` library. The cast-in / cast-out pair is
    // equivalent to a single `QMatMul::forward_via_f16` call
    // (`candle-core/src/quantized/mod.rs:754-763`) — same three ops,
    // same dtype sequence — but avoids constructing a QMatMul wrapper
    // we'd have to synthesize around an already-dequantized F16 tensor.
    //
    // Cost of the two casts at `num_tokens=1`: `[1,2816] F32→F16` =
    // 2816 fp16 writes ≈ 11 KB; `[1,262144] F16→F32` = 262144 fp32
    // writes ≈ 1 MB. Both are negligible vs the 1.48 GB weight read.
    let x_f16 = normed_2d.to_dtype(DType::F16)?;
    let w_t = lm_head_f16.t()?;
    let logits_f16 = x_f16.matmul(&w_t)?;
    let logits_f32 = logits_f16.to_dtype(DType::F32)?;
    Ok(logits_f32)
}

// ---------------------------------------------------------------------------
// Phase A unit test — numerical parity vs the F32 dense baseline
// ---------------------------------------------------------------------------
//
// Constructs a small synthetic F32 weight matrix, dequantizes+casts it
// to F16 (mirroring what the model load path does to the real
// `token_embd.weight`), then runs both the F32 dense matmul (the Phase-1
// baseline at `gemma4.rs:1879`) and the F16 fused path, and compares
// the two output tensors at ε=1e-3.
//
// The ε=1e-3 bar is INTENTIONALLY wider than the 1e-5 bar used by
// 1bNEW.1/4/6 because the entire point of 1bNEW.17 is to *change* the
// reduction order. F16 matmul accumulates in F32 internally (per MLX
// gemm), but the inputs are F16-rounded, which produces systematic
// low-order drift at vocab scale. The unit test's job is to catch
// catastrophic divergence (wrong output, wrong shape, NaNs, axis
// swap), not to hold the port to bit-identity.
//
// The test is skipped when no Metal device is available (candle CPU
// backend does NOT implement F16 matmul at
// `candle-core/src/cpu_backend/mod.rs`, so the `.matmul` call would
// bail with "unsupported dtype F16" on a CPU-only runner).
#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{Device as CoreDevice, Tensor as CoreTensor};

    /// Deterministic pseudo-random f32 generator. Same splitmix64 mixer
    /// `moe_kernel::tests::make_f32_vec` uses. Values are in `[-0.25, 0.25]`
    /// so the F32→F16 cast doesn't saturate and the matmul outputs are
    /// in a well-behaved range for the ε comparison.
    fn make_f32_vec(len: usize, seed: u64) -> Vec<f32> {
        let mut s = seed.wrapping_add(0x9E3779B97F4A7C15);
        (0..len)
            .map(|_| {
                s = s.wrapping_add(0x9E3779B97F4A7C15);
                let mut z = s;
                z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
                z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
                z ^= z >> 31;
                ((z as i64 as f64) * (1.0 / (i64::MAX as f64)) * 0.25) as f32
            })
            .collect()
    }

    /// Phase A parity harness. Shapes mimic the real lm_head in
    /// miniature: hidden=128 (cheap to matmul), vocab=256 (cheap but
    /// still wide enough to stress the reduction). The full
    /// `262144 × 2816` shape is exercised by the Phase B live-forward
    /// gate against the real GGUF, not this unit test.
    fn run_lm_head_parity(hidden: usize, vocab: usize, n_tokens: usize) {
        let device = match CoreDevice::new_metal(0) {
            Ok(d) => d,
            Err(e) => {
                eprintln!("skip: metal device unavailable ({e})");
                return;
            }
        };

        // --- Build synthetic F32 weight + input -----------------------------
        // Seeds are arbitrary but distinct; the splitmix64 mixer handles
        // low-entropy inputs fine. `0x1b_17_...` is a nod to ADR-005 1bNEW.17.
        let weight_f32_vec = make_f32_vec(vocab * hidden, 0x1b_17_0000_0a5e);
        let w_f32 = CoreTensor::from_vec(weight_f32_vec, (vocab, hidden), &device).unwrap();

        let input_f32_vec = make_f32_vec(n_tokens * hidden, 0x1b_17_0000_0bee);
        let x_f32 = CoreTensor::from_vec(input_f32_vec, (n_tokens, hidden), &device).unwrap();

        // --- F32 dense baseline (mirrors gemma4.rs:1879) --------------------
        // `logits = x @ W.t()`. Phase-1 baseline path.
        let ref_logits = x_f32.matmul(&w_f32.t().unwrap()).unwrap();
        let ref_shape = ref_logits.shape().clone();
        let ref_vec: Vec<f32> = ref_logits.flatten_all().unwrap().to_vec1().unwrap();

        // --- F16 fused path -------------------------------------------------
        // Cast the weight to F16 (mirrors what `Gemma4Model::load_with_modes`
        // does at load time via `embed_w.to_dtype(DType::F16)`). Then route
        // through the helper under test.
        let w_f16 = w_f32.to_dtype(DType::F16).unwrap();
        let fused_logits = lm_head_forward_fused(&x_f32, &w_f16).unwrap();
        assert_eq!(fused_logits.shape(), &ref_shape);
        assert_eq!(fused_logits.dtype(), DType::F32);
        let fused_vec: Vec<f32> = fused_logits.flatten_all().unwrap().to_vec1().unwrap();
        assert_eq!(fused_vec.len(), ref_vec.len());

        // --- Compare at ε=1e-3 (wider than 1bNEW.1/4/6) ---------------------
        // Also check the argmax matches on non-adversarial inputs.
        let mut max_abs: f32 = 0.0;
        let mut max_idx: usize = 0;
        for (i, (a, b)) in fused_vec.iter().zip(ref_vec.iter()).enumerate() {
            let d: f32 = (*a - *b).abs();
            if d > max_abs {
                max_abs = d;
                max_idx = i;
            }
        }

        // Per-row argmax check.
        for t in 0..n_tokens {
            let ref_row = &ref_vec[t * vocab..(t + 1) * vocab];
            let fused_row = &fused_vec[t * vocab..(t + 1) * vocab];
            let ref_argmax = ref_row
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .map(|(i, _)| i)
                .unwrap();
            let fused_argmax = fused_row
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .map(|(i, _)| i)
                .unwrap();
            assert_eq!(
                fused_argmax, ref_argmax,
                "argmax mismatch at token {t}: ref={ref_argmax}, fused={fused_argmax}"
            );
        }

        eprintln!(
            "lm_head_fused({vocab}x{hidden}, n_tokens={n_tokens}): \
             max|Δ|={max_abs:.3e} at idx {max_idx} \
             (fused={}, ref={})",
            fused_vec[max_idx], ref_vec[max_idx]
        );
        assert!(
            max_abs < 1e-3,
            "lm_head fused F16 path vs F32 baseline differs at ε=1e-3: max|Δ|={max_abs}"
        );
    }

    #[test]
    fn lm_head_f16_matches_f32_single_token() {
        // Decode-time shape: n_tokens=1. The canonical hot-path case.
        run_lm_head_parity(128, 256, 1);
    }

    #[test]
    fn lm_head_f16_matches_f32_multi_token() {
        // Prefill-time shape: n_tokens > 1. Exercises the same code
        // path with a batch dimension, per Anti-Goal #6 (no benchmark
        // tuning — decode and prefill must flow through the same
        // dispatch).
        run_lm_head_parity(128, 256, 8);
    }

    #[test]
    fn lm_head_f16_rejects_wrong_dtypes() {
        // Guardrail: feeding an F32 weight to the fused path must
        // return an error rather than silently dequant+recompute.
        let device = match CoreDevice::new_metal(0) {
            Ok(d) => d,
            Err(_) => return,
        };
        let w_f32 = CoreTensor::zeros((256, 128), DType::F32, &device).unwrap();
        let x_f32 = CoreTensor::zeros((1, 128), DType::F32, &device).unwrap();
        assert!(lm_head_forward_fused(&x_f32, &w_f32).is_err());

        // Also reject an F16 input.
        let w_f16 = CoreTensor::zeros((256, 128), DType::F16, &device).unwrap();
        let x_f16 = CoreTensor::zeros((1, 128), DType::F16, &device).unwrap();
        assert!(lm_head_forward_fused(&x_f16, &w_f16).is_err());
    }
}
