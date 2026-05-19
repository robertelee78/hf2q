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

    /// Phase B (forward-pass interception) is not yet shipped. Used when
    /// `--imatrix-corpus <name>` is requested but hf2q's in-tree
    /// imatrix generator is still on the TODO list (the Phase A
    /// deliverable for ADR-033 §Pi).
    #[error(
        "imatrix: in-tree imatrix generation (Phase B forward-pass driver) is not yet \
         shipped (ADR-033 §Pi). Workaround: generate a `.imatrix.gguf` with stock \
         `llama-imatrix -m <gguf> -f data/calibration/{corpus}.txt -o <out>.imatrix.gguf` \
         and pass it via `--imatrix <out>.imatrix.gguf`."
    )]
    InTreeGenerationNotYetShipped { corpus: String },

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
