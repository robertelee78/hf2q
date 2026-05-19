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
}
