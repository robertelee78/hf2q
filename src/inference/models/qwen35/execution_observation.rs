//! Exact loaded and executed tensor observations for dense-Qwen evidence.

mod executed;
mod loaded;

pub(super) use executed::ExecutedTensorCatalogBuilder;
pub(crate) use executed::{ExecutedTensorObservation, VerifiedExecutedTensorCatalog};
pub(super) use loaded::{capture_loaded_tensor_catalog, observe_loaded_f32, observe_loaded_ggml};
pub(crate) use loaded::{LoadedTensorCodec, VerifiedLoadedTensorCatalog};
