//! Forward-pass driver — STUB for Phase B.
//!
//! ## Status: DEFERRED to Phase B
//!
//! Per ADR-033 §Pi the forward-pass driver runs hf2q's existing decoder
//! forward-pass (`src/inference/models/<arch>/`) over the calibration
//! corpus chunks, intercepting the input activations to each linear
//! layer being quantized. This is the load-bearing piece of the imatrix
//! subsystem — it requires reading every per-arch decoder's
//! `MlxAffineLinear::forward` (or equivalent) call site and threading
//! an opt-in `ImatrixCollector` hook through them.
//!
//! ## Why deferred
//!
//! The interception is invasive: each of the 5 inference-supported
//! arches (`Bert`, `Gemma4`, `NomicBert`, `Qwen35Moe`, `Qwen3VlText`)
//! has its own decoder forward pass with arch-specific MoE routing,
//! attention fusion, and KV cache handling. Adding the hook safely
//! without disrupting the live serve path (per the §Pi constraint
//! "don't break existing inference paths") requires a careful
//! per-arch audit that is too large for a single Phase A iter.
//!
//! ## Phase A workaround (operator-visible)
//!
//! Operators who need an `.imatrix.gguf` today can:
//!
//! ```bash
//! # 1. Convert the HF model to a F16/BF16 GGUF first.
//! hf2q convert-v2 <hf-dir> --quant f16 -o /tmp/m-f16.gguf
//!
//! # 2. Generate the imatrix via stock llama-imatrix.
//! llama-imatrix \
//!     -m /tmp/m-f16.gguf \
//!     -f /opt/hf2q/data/calibration/cdv3.txt \
//!     -o /tmp/m.imatrix.gguf \
//!     --output-format gguf -ngl 999
//!
//! # 3. Pass the resulting imatrix back to hf2q for I-tier APEX.
//! hf2q convert-v2 <hf-dir> --quant apex-i-balanced \
//!     --imatrix /tmp/m.imatrix.gguf \
//!     -o /tmp/m-apex-i-balanced.gguf
//! ```
//!
//! This Phase A path satisfies the §Pi acceptance gate (operators get
//! working I-tier APEX) without the deep forward-pass interception
//! Phase B work.
//!
//! ## Phase B contract sketch
//!
//! Phase B will introduce an `ImatrixCollector` trait that the per-arch
//! decoder forward-pass code can call at each linear-input site:
//!
//! ```ignore
//! pub trait ImatrixCollector {
//!     /// Called by `MlxAffineLinear::forward` BEFORE matmul.
//!     /// `name` is the canonical GGUF tensor name; `activations`
//!     /// is the per-token input slice (n_tokens × n_per_row).
//!     fn collect_dense(&mut self, name: &str, activations: &[f32], n_tokens: usize);
//!     /// Called by MoE expert dispatch. `expert_id` is the chosen
//!     /// expert slot; activations are the row(s) routed to it.
//!     fn collect_moe(&mut self, name: &str, expert_id: usize, activations: &[f32]);
//! }
//! ```
//!
//! At convert time the driver constructs a [`super::accumulator::AccumulatorRegistry`]-backed
//! `ImatrixCollector` impl and runs the per-arch decoder over the
//! tokenized corpus. The accumulators are then handed to
//! [`super::gguf_writer::write_imatrix`] for serialization.

use std::path::PathBuf;

use super::corpus::CorpusBytes;
use super::error::ImatrixError;
use crate::quantize::ggml_quants::ArchName;

/// Driver-side parameters for an in-tree imatrix run.
///
/// Phase A: parsed by the CLI but not yet consumed — passing this to
/// [`compute_imatrix`] returns [`ImatrixError::InTreeGenerationNotYetShipped`].
#[derive(Debug, Clone)]
pub struct ComputeImatrixParams {
    /// HF model directory (config.json + safetensors).
    pub hf_dir: PathBuf,
    /// Corpus text payload.
    pub corpus: CorpusBytes,
    /// `n_ctx` used by the forward pass. `chunk_size = n_ctx / n_parallel`
    /// per ADR-033 §Pi (default `n_parallel = 1` ⇒ chunks the corpus
    /// into `n_ctx`-token windows).
    pub n_ctx: u32,
    /// Detected source arch (gemma4 / qwen35moe / etc.).
    pub arch: ArchName,
}

/// Phase B entry point. Phase A returns
/// [`ImatrixError::InTreeGenerationNotYetShipped`] verbatim.
pub fn compute_imatrix(params: &ComputeImatrixParams) -> Result<(), ImatrixError> {
    // Phase B note for the implementer: at this point you have
    // (hf_dir, corpus, n_ctx, arch). Tokenize `params.corpus` via the
    // per-arch tokenizer (same code-path as
    // `src/convert/tokenizer.rs::build_tokenizer_metadata` but at
    // token-id resolution time, not metadata-emission time). Then
    // chunk via `super::corpus::chunk_tokens(&tokens, n_ctx / 1)`. For
    // each chunk, run hf2q's per-arch decoder forward pass with an
    // attached `ImatrixCollector` (see module-doc trait sketch above).
    // Finally call `super::gguf_writer::write_imatrix(...)` on the
    // resulting `AccumulatorRegistry`.
    Err(ImatrixError::InTreeGenerationNotYetShipped {
        corpus: params.corpus.label.clone(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::quantize::imatrix::corpus::CorpusSource;

    /// Phase A: calling `compute_imatrix` returns the deferred error,
    /// not a panic / silent no-op. Composes with the
    /// no-loop-suppression rule.
    #[test]
    fn compute_imatrix_returns_deferred_error() {
        let corpus = CorpusBytes::load(&CorpusSource::Cdv3).unwrap();
        let params = ComputeImatrixParams {
            hf_dir: PathBuf::from("/tmp/non-existent-fixture"),
            corpus,
            n_ctx: 512,
            arch: ArchName::Gemma4,
        };
        let err = compute_imatrix(&params).unwrap_err();
        match err {
            ImatrixError::InTreeGenerationNotYetShipped { corpus } => {
                assert_eq!(corpus, "cdv3");
            }
            other => panic!("expected InTreeGenerationNotYetShipped, got {other:?}"),
        }
    }
}
