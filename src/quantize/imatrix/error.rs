//! Typed errors for the `src/quantize/imatrix/` subsystem.
//!
//! Per ADR-033 §Pi + [[feedback-no-loop-suppression-2026-05-17]]:
//! every failure mode is a typed variant — never a silent fallback to
//! "imatrix-less quantize". A missing tensor in a loaded `.imatrix.gguf`,
//! a malformed corpus, or a schema-violating GGUF surfaces here.

use thiserror::Error;

/// Typed errors raised by the imatrix subsystem.
#[derive(Error, Debug)]
pub enum ImatrixError {
    /// Underlying I/O failure (file open / read / write / seek).
    #[error("imatrix I/O: {0}")]
    Io(#[from] std::io::Error),

    /// Underlying GGUF writer failure (from
    /// [`crate::backends::gguf::writer::WriterError`]).
    #[error("imatrix gguf writer: {0}")]
    Writer(#[from] crate::backends::gguf::writer::WriterError),

    /// The on-disk file does not parse as a GGUF (bad magic / version /
    /// truncated header). Wrapped from `mlx_native::gguf::GgufFile::open`.
    #[error("imatrix gguf parse: {detail}")]
    Parse { detail: String },

    /// The on-disk file parsed as a GGUF but is not an imatrix file —
    /// `general.type` is absent or not `"imatrix"`. Per the
    /// no-silent-fallback rule we refuse to consume a non-imatrix GGUF
    /// as if it were one.
    #[error(
        "imatrix file `{path}` is not an imatrix GGUF (general.type=`{actual}`); \
         expected general.type=`imatrix`"
    )]
    NotAnImatrix { path: String, actual: String },

    /// The file is missing one of the canonical KV pairs required by
    /// the llama-imatrix v3 schema (`imatrix.datasets`, `imatrix.chunk_count`,
    /// `imatrix.chunk_size`).
    #[error("imatrix file `{path}` missing required key `{key}`")]
    MissingKv { path: String, key: &'static str },

    /// A tensor pair `<name>.in_sum2 / <name>.counts` was incomplete —
    /// either `in_sum2` exists without `counts` or vice versa. Per
    /// llama-imatrix's `load_imatrix` (imatrix.cpp:783-788) this is a
    /// hard error there too.
    #[error(
        "imatrix file `{path}` tensor `{name}`: \
         in_sum2 and counts must both be present"
    )]
    MismatchedTensorPair { path: String, name: String },

    /// Corpus loader: the supplied path could not be read as a UTF-8
    /// text file.
    #[error("imatrix corpus `{path}`: {detail}")]
    CorpusRead { path: String, detail: String },

    /// Corpus loader: the requested baked corpus name is not one of the
    /// canonical baked corpora.
    #[error(
        "imatrix: unknown baked corpus `{name}`; \
         supported baked corpora: {supported:?}"
    )]
    UnknownBakedCorpus {
        name: String,
        supported: &'static [&'static str],
    },

    /// Phase B intercept: the materialized input buffer doesn't hold
    /// `m * n_per_row` F32 values. Indicates a wiring bug in the
    /// `dispatch_qmatmul` callsite (wrong `m` argument, wrong
    /// `weight.info.cols`, or a buffer allocated with extra padding
    /// past valid data). Per [[feedback-no-loop-suppression-2026-05-17]]
    /// this is a typed error, not a silent skip — silently dropping
    /// activation data biases the imatrix output.
    #[error(
        "imatrix intercept: buffer/shape mismatch for `{tensor}`: \
         got {got} f32 values, expected m*n_per_row = {m}*{n_per_row} = {expected}"
    )]
    ShapeMismatch {
        tensor: String,
        m: usize,
        n_per_row: usize,
        got: usize,
        expected: usize,
    },

    /// Phase B Stage 3 driver: `run_convert` failed when converting
    /// the source HF directory to a temporary F16 GGUF. Wraps the
    /// upstream `ConvertError` debug-formatted (avoids a circular
    /// dependency on `ConvertError` from this module).
    #[error("imatrix driver: convert to F16 GGUF failed: {detail}")]
    ConvertFailed { detail: String },

    /// Phase B Stage 3 driver: `LoadedModel::load` failed on the
    /// temporary F16 GGUF. Wraps the upstream `anyhow::Error`
    /// debug-formatted.
    #[error("imatrix driver: model load failed: {detail}")]
    ModelLoadFailed { detail: String },

    /// Phase B Stage 3 driver: the source arch isn't yet wired for
    /// in-tree imatrix generation. Listed arches have the inference
    /// path (per-arch decoder + tokenizer) the driver needs;
    /// everything else is an explicit-error opt-in code change.
    #[error(
        "imatrix driver: arch `{arch}` is not yet wired for in-tree imatrix generation; \
         supported arches: {supported:?}"
    )]
    UnsupportedArchForDriver {
        arch: String,
        supported: &'static [&'static str],
    },

    /// Phase B Stage 3 driver: tokenization of the calibration corpus
    /// failed (HF tokenizers crate error). Wraps the upstream error
    /// debug-formatted.
    #[error("imatrix driver: tokenizer encode failed: {detail}")]
    TokenizationFailed { detail: String },

    /// Phase B Stage 3 driver: `forward_prefill` failed mid-chunk.
    /// `chunk_index` is 0-based, `chunk_count` is the total number of
    /// chunks the driver planned to walk. Wraps the upstream
    /// `anyhow::Error`.
    #[error(
        "imatrix driver: forward pass failed on chunk {chunk_index}/{chunk_count}: {detail}"
    )]
    ForwardPassFailed {
        chunk_index: usize,
        chunk_count: usize,
        detail: String,
    },

    /// Phase B Stage 3 driver: the corpus tokenized into too few
    /// tokens to fill even one chunk of size `n_ctx`. Per the
    /// canonical llama-imatrix behavior at `imatrix.cpp:960` partial
    /// trailing chunks are dropped; an all-trailing corpus produces
    /// zero chunks and an empty imatrix is meaningless.
    #[error(
        "imatrix driver: corpus `{corpus_label}` tokenized to {token_count} tokens, \
         insufficient for even one chunk of size {n_ctx}"
    )]
    CorpusTooShort {
        corpus_label: String,
        token_count: usize,
        n_ctx: u32,
    },
}

#[cfg(test)]
mod p7_ac3_hint_tests {
    //! ADR-033 §P7 AC#3 — every typed-error MESSAGE must carry the
    //! actionable hint. Variant-only `matches!()` checks don't catch
    //! hint regressions, so each variant's `.to_string()` is
    //! substring-checked here.

    use super::*;

    #[test]
    fn p7_ac3_not_an_imatrix_lists_expectation() {
        let msg = ImatrixError::NotAnImatrix {
            path: "/tmp/foo.gguf".to_string(),
            actual: "llama".to_string(),
        }
        .to_string();
        assert!(msg.contains("/tmp/foo.gguf"), "msg should echo path: {msg}");
        assert!(msg.contains("imatrix"), "msg should reference imatrix: {msg}");
        assert!(msg.contains("llama"), "msg should report actual general.type: {msg}");
    }

    #[test]
    fn p7_ac3_missing_kv_names_field() {
        let msg = ImatrixError::MissingKv {
            path: "/tmp/x.gguf".to_string(),
            key: "imatrix.chunk_count",
        }
        .to_string();
        assert!(msg.contains("imatrix.chunk_count"), "msg should name the key: {msg}");
        assert!(msg.contains("/tmp/x.gguf"), "msg should echo path: {msg}");
    }

    #[test]
    fn p7_ac3_unknown_baked_corpus_lists_supported() {
        let msg = ImatrixError::UnknownBakedCorpus {
            name: "wikitext-9000".to_string(),
            supported: &["cdv3", "mudler", "user-file"],
        }
        .to_string();
        assert!(msg.contains("wikitext-9000"), "msg should echo bad name: {msg}");
        assert!(msg.contains("cdv3"), "msg should list a supported value: {msg}");
        assert!(
            msg.contains("user-file"),
            "msg should mention the operator-supplied option: {msg}"
        );
    }

    #[test]
    fn p7_ac3_shape_mismatch_carries_diagnostic_dims() {
        let msg = ImatrixError::ShapeMismatch {
            tensor: "blk.0.attn_q.weight".to_string(),
            m: 2,
            n_per_row: 4,
            got: 5,
            expected: 8,
        }
        .to_string();
        assert!(msg.contains("blk.0.attn_q.weight"), "msg should name tensor: {msg}");
        assert!(msg.contains("2"), "msg should carry m: {msg}");
        assert!(msg.contains("4"), "msg should carry n_per_row: {msg}");
        assert!(msg.contains("5"), "msg should carry got: {msg}");
        assert!(msg.contains("8"), "msg should carry expected: {msg}");
    }

    #[test]
    fn p7_ac3_convert_failed_carries_detail() {
        let msg = ImatrixError::ConvertFailed {
            detail: "hf_dir `/tmp/bogus` does not exist".to_string(),
        }
        .to_string();
        assert!(
            msg.contains("convert to F16 GGUF failed"),
            "msg should name the failed step: {msg}"
        );
        assert!(msg.contains("/tmp/bogus"), "msg should carry upstream detail: {msg}");
    }

    #[test]
    fn p7_ac3_model_load_failed_carries_detail() {
        let msg = ImatrixError::ModelLoadFailed {
            detail: "GGUF magic mismatch".to_string(),
        }
        .to_string();
        assert!(msg.contains("model load failed"), "msg should name the step: {msg}");
        assert!(msg.contains("GGUF magic mismatch"), "msg should carry detail: {msg}");
    }

    #[test]
    fn p7_ac3_unsupported_arch_for_driver_lists_supported() {
        let msg = ImatrixError::UnsupportedArchForDriver {
            arch: "qwen3moe".to_string(),
            supported: &["gemma4"],
        }
        .to_string();
        assert!(msg.contains("qwen3moe"), "msg should echo bad arch: {msg}");
        assert!(msg.contains("gemma4"), "msg should list supported arches: {msg}");
        assert!(
            msg.contains("not yet wired") || msg.contains("supported"),
            "msg should explain the gap: {msg}"
        );
    }

    #[test]
    fn p7_ac3_tokenization_failed_carries_detail() {
        let msg = ImatrixError::TokenizationFailed {
            detail: "Encoding error: invalid UTF-8 at byte 42".to_string(),
        }
        .to_string();
        assert!(msg.contains("tokenizer"), "msg should name the step: {msg}");
        assert!(msg.contains("byte 42"), "msg should carry upstream detail: {msg}");
    }

    #[test]
    fn p7_ac3_forward_pass_failed_locates_chunk() {
        let msg = ImatrixError::ForwardPassFailed {
            chunk_index: 7,
            chunk_count: 100,
            detail: "GPU OOM".to_string(),
        }
        .to_string();
        assert!(
            msg.contains("7") && msg.contains("100"),
            "msg should locate chunk_index/chunk_count: {msg}"
        );
        assert!(msg.contains("GPU OOM"), "msg should carry upstream detail: {msg}");
    }

    #[test]
    fn p7_ac3_corpus_too_short_carries_dims() {
        let msg = ImatrixError::CorpusTooShort {
            corpus_label: "user-file:tiny.txt".to_string(),
            token_count: 32,
            n_ctx: 512,
        }
        .to_string();
        assert!(msg.contains("user-file:tiny.txt"), "msg should echo label: {msg}");
        assert!(msg.contains("32"), "msg should carry token_count: {msg}");
        assert!(msg.contains("512"), "msg should carry n_ctx: {msg}");
    }
}
