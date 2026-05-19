//! Per-architecture HF→GGUF tensor-name mappers + metadata builders.
//!
//! ADR-033 §P0 "Per-arch convert-side mapping at `src/convert/arch/<arch>.rs`".
//! v1 ships only `llama3` (dense decoder test fixture for the convert
//! matrix); extending to `{gemma4, qwen35moe, qwen3vl, gemma4_mmproj,
//! bert, nomic_bert, minimax_m2}` is iter-23+ work per the ADR.
//!
//! Per [[feedback-no-backwards-compat-2026-05-18]]: no compat shims, no
//! per-arch fallback. Adding a new arch is an explicit code change that
//! adds a new file under this module — `ArchName` is closed enum.
//!
//! Per [[feedback-no-loop-suppression-2026-05-17]]: `map_tensor_name`
//! returning `None` is a signal the caller MUST surface (e.g. as
//! `ConvertError::UnmappedTensor { hf_name }`). Never silently skip.

pub mod llama3;
