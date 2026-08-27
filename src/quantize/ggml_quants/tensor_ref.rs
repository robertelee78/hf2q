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
    /// DeepSeek-V4 routed expert weights: two E2M1 values packed into
    /// each I8/U8 byte and scaled in groups of 32 by E8M0.
    Mxfp4E2M1,
    /// E8M0 sidecar scale indexed by the reader but never emitted.
    E8M0Scale,
    /// Integer hash-routing table preserved as GGML I32.
    I32,
    /// I64 source route table range-checked into canonical GGML I32.
    I64,
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
///    the peer's. The convert pipeline doesn't yet support
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
    /// Google Gemma 4 multimodal projector (mmproj sidecar) —
    /// **Gemma 3 SigLIP variant**. Tensor naming
    /// `model.vision_tower.vision_model.encoder.layers.<N>.*` with the
    /// SigLIP-style block (q_proj, k_proj, mlp.fc1, mlp.fc2,
    /// layer_norm1, layer_norm2). Handles the Gemma 3 4B/12B vision
    /// release. For the Gemma 4 26B-A4B-IT vision tower (transformer-
    /// style with SwiGLU + per-head q/k_norm) see
    /// [`ArchName::Gemma4VisionMmproj`].
    Gemma4Mmproj,
    /// Google Gemma 4 vision mmproj — **transformer-style variant**
    /// shipped with the 26B-A4B-IT release. Tensor naming
    /// `model.vision_tower.encoder.layers.<N>.*` (plural `layers`, no
    /// `vision_model` infix), block has SwiGLU FFN + post-attn/post-ffn
    /// norms + per-head q/k_norm. Projector type 'gemma4v'. Canonical
    /// handler: the peer's `Gemma4VisionAudioModel`.
    Gemma4VisionMmproj,
    /// Dense Qwen3.5-family hybrid decoder (`qwen35` GGUF), including
    /// Qwen3.8-27B. This is distinct from both Qwen MoE variants.
    Qwen35,
    /// Qwen 3.5/3.6 MoE-A3B family — older dense-MoE variant (gguf
    /// upstream label `qwen3moe`, no linear-attention, no MTP heads,
    /// no shared experts). Used for HF arches like `Qwen3MoeForCausalLM`.
    /// See [`ArchName::Qwen35MoeFull`] for the newer Qwen 3.5
    /// multimodal/MTP variant.
    Qwen35Moe,
    /// Qwen 3.5 MoE-A3B with linear-attention + MTP heads (gguf
    /// upstream label `qwen35moe`, gguf-py `MODEL_ARCH.QWEN35MOE`).
    /// Handles `Qwen3_5MoeForConditionalGeneration` (multimodal-wrapping
    /// `model.language_model.*` + `model.visual.*`) and
    /// `Qwen3_5MoeForCausalLM` (text-only). Canonical handler:
    /// the peer's `Qwen3_5MoeTextModel`
    /// (inherits `_Qwen35MtpMixin`, `_Qwen35MRopeMixin`,
    /// `_LinearAttentionVReorderBase`).
    Qwen35MoeFull,
    /// BERT family (BAAI bge-large-en, etc.).
    Bert,
    /// Nomic-BERT embedding model family.
    NomicBert,
    /// Llama-3 dense decoder (8B test fixture for convert matrix).
    Llama3,
    /// MiniMax M2.7 (FP8 source).
    MiniMaxM2,
    /// DeepSeek-V4 hybrid-compressed-attention MoE decoder.
    Deepseek4,

    // --- C-fidelity placeholders for `target_for` (no convert support yet) ---
    /// Falcon — explicitly checked in 6 places inside
    /// `llama_tensor_get_type_impl`.
    Falcon,
}

impl ArchName {
    /// Canonical lowercase name used in GGUF metadata
    /// (`general.architecture`) and in error messages.
    pub const fn name(self) -> &'static str {
        match self {
            ArchName::Gemma4 => "gemma4",
            ArchName::Gemma4Mmproj => "gemma4_mmproj",
            ArchName::Gemma4VisionMmproj => "gemma4_vision_mmproj",
            ArchName::Qwen35 => "qwen35",
            ArchName::Qwen35Moe => "qwen3moe",
            ArchName::Qwen35MoeFull => "qwen35moe",
            ArchName::Bert => "bert",
            ArchName::NomicBert => "nomic-bert",
            ArchName::Llama3 => "llama",
            ArchName::MiniMaxM2 => "minimax-m2",
            ArchName::Deepseek4 => "deepseek4",
            ArchName::Falcon => "falcon",
        }
    }

    /// Inverse of [`Self::name`] — parse a canonical lowercase arch
    /// label (the form stored in `general.architecture` and in
    /// `data/apex-references/manifest.json::entries[].arch`) back into
    /// an [`ArchName`].
    ///
    /// Returns `None` for unknown labels. Per
    /// [[feedback-no-backwards-compat-2026-05-18]] adding a new arch
    /// is an explicit code change — there's no implicit aliasing or
    /// stem-stripping.
    pub fn from_label(label: &str) -> Option<Self> {
        match label {
            "gemma4" => Some(ArchName::Gemma4),
            "gemma4_mmproj" => Some(ArchName::Gemma4Mmproj),
            "gemma4_vision_mmproj" => Some(ArchName::Gemma4VisionMmproj),
            "qwen35" => Some(ArchName::Qwen35),
            // Two explicit labels for the same arch:
            //   - "qwen3moe" — upstream GGUF metadata convention
            //     (`general.architecture` value).
            //   - "qwen35moe" — hf2q-internal label used in
            //     `data/apex-references/manifest.json` (distinguishes
            //     the operator's qwen3.5/3.6 family from a pure qwen3
            //     series for fingerprint-routing purposes).
            // Both are explicit code-level entries per the "no
            // implicit aliasing / migration" rule.
            // "qwen3moe" — older dense-MoE upstream label
            "qwen3moe" => Some(ArchName::Qwen35Moe),
            // "qwen35moe" — newer linear-attn + MTP variant
            "qwen35moe" => Some(ArchName::Qwen35MoeFull),
            "bert" => Some(ArchName::Bert),
            "nomic-bert" => Some(ArchName::NomicBert),
            "llama" => Some(ArchName::Llama3),
            "minimax-m2" => Some(ArchName::MiniMaxM2),
            "deepseek4" => Some(ArchName::Deepseek4),
            "falcon" => Some(ArchName::Falcon),
            _ => None,
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
            ArchName::Qwen35,
            ArchName::Qwen35Moe,
            ArchName::Bert,
            ArchName::Llama3,
            ArchName::MiniMaxM2,
        ] {
            assert!(
                arch.name()
                    .chars()
                    .all(|c| c.is_ascii_lowercase() || c == '_' || c == '-' || c.is_ascii_digit()),
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
