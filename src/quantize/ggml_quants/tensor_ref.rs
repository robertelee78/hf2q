//! `TensorRef` — input to `QuantPolicy::target_for`.
//!
//! Per ADR-033 Decision §"TensorRef (passed to QuantPolicy::target_for)".
//! Carries everything `llama_tensor_get_type_impl` and `tensor_type_fallback`
//! need to decide a `GgmlType` for one tensor — name, shape, source
//! dtype, arch, and (for per-block tensors) layer index.

/// Source dtype of the safetensors-side tensor before
/// conversion. Per ADR Decision §"FP8 source-dtype auto-detect"
/// the FP8 path is auto-detected; `Fp8E4M3` covers MiniMax-M2.7
/// and others with `quantization_config.quant_method == "fp8"`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SourceDtype {
    F32,
    F16,
    BF16,
    /// `float8_e4m3fn` — 1-bit sign + 4-bit exp + 3-bit mantissa, no
    /// inf, single NaN encoding. Auto-dequantized to F32 in-memory.
    Fp8E4M3,
}

/// Closed enum of architectures recognized by `StandardPolicy::target_for`.
///
/// Two tiers:
///
/// 1. **v1 convert arches** (Gemma4/Qwen35Moe/Bert/NomicBert/Llama3/
///    MiniMaxM2 and their multimodal siblings) — first-class for the
///    convert pipeline.
/// 2. **C-fidelity placeholders** — `Falcon` and the rest exist so
///    `target_for` can express its arch-keyed branches verbatim against
///    `llama-quant.cpp`. The convert pipeline doesn't yet support
///    quantizing models of these architectures; they show up in
///    `target_for` only because the C function branches on them.
///
/// Per [[feedback-no-backwards-compat-2026-05-18]]: there's no implicit
/// detection / migration — adding an arch is an explicit code change.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ArchName {
    // --- v1 production arches ---
    /// Google Gemma 4 (Gemma3-architecture compatible).
    Gemma4,
    /// Google Gemma 4 multimodal projector (mmproj sidecar).
    Gemma4Mmproj,
    /// Qwen 3.5/3.6 MoE-A3B family (qwen3moe upstream).
    Qwen35Moe,
    /// Qwen3-VL text-side decoder.
    Qwen3VlText,
    /// BERT family (BAAI bge-large-en, etc.).
    Bert,
    /// Nomic-BERT embedding model family.
    NomicBert,
    /// Llama-3 dense decoder (8B test fixture for convert matrix).
    Llama3,
    /// MiniMax M2.7 (FP8 source).
    MiniMaxM2,

    // --- C-fidelity placeholders for `target_for` (no convert support yet) ---
    /// Falcon — explicitly checked in 6 places inside
    /// `llama_tensor_get_type_impl` (L449, L580, L588, L591, L602, L614).
    Falcon,
}

impl ArchName {
    /// Canonical lowercase name used in GGUF metadata
    /// (`general.architecture`) and in error messages.
    pub const fn name(self) -> &'static str {
        match self {
            ArchName::Gemma4 => "gemma4",
            ArchName::Gemma4Mmproj => "gemma4_mmproj",
            ArchName::Qwen35Moe => "qwen3moe",
            ArchName::Qwen3VlText => "qwen3vl",
            ArchName::Bert => "bert",
            ArchName::NomicBert => "nomic-bert",
            ArchName::Llama3 => "llama",
            ArchName::MiniMaxM2 => "minimax-m2",
            ArchName::Falcon => "falcon",
        }
    }
}

/// Per-tensor reference passed into `QuantPolicy::target_for`.
#[derive(Debug, Clone, Copy)]
pub struct TensorRef<'a> {
    /// Canonical GGUF tensor name (e.g. `"blk.0.attn_q.weight"`).
    pub name: &'a str,
    /// Row-major dimensions. `shape[0]` is `n_per_row` (the inner dim
    /// the quantizer iterates over per row).
    pub shape: &'a [usize],
    /// safetensors source dtype.
    pub source_dtype: SourceDtype,
    /// Architecture this tensor belongs to.
    pub arch: ArchName,
    /// `None` for global tensors (`token_embd`, `output`, etc.);
    /// `Some(i)` for `blk.<i>.*` per-block tensors.
    pub layer_index: Option<usize>,
}

impl<'a> TensorRef<'a> {
    /// `n_per_row` — the inner dimension the quantizer iterates over.
    pub const fn n_per_row(&self) -> usize {
        self.shape[0]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn arch_names_lowercase() {
        for arch in [
            ArchName::Gemma4,
            ArchName::Qwen35Moe,
            ArchName::Bert,
            ArchName::Llama3,
            ArchName::MiniMaxM2,
        ] {
            assert!(
                arch.name().chars().all(|c| c.is_ascii_lowercase() || c == '_' || c == '-' || c.is_ascii_digit()),
                "arch name {} must be lowercase/digits/_-",
                arch.name()
            );
        }
    }

    #[test]
    fn tensor_ref_n_per_row() {
        let shape = [4096, 32];
        let t = TensorRef {
            name: "blk.0.attn_q.weight",
            shape: &shape,
            source_dtype: SourceDtype::BF16,
            arch: ArchName::Llama3,
            layer_index: Some(0),
        };
        assert_eq!(t.n_per_row(), 4096);
    }
}
