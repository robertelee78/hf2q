//! Source-to-runtime physical tensor provenance.
//!
//! Quantization policy is not runtime truth. A stored GGUF tensor may be
//! dequantized, split, transposed, padded, or requantized before an inference
//! operation consumes it. This module records that physical DAG and verifies
//! it independently before Dynamic allocation or Apple cost evidence may bind
//! to it.

mod types;
mod verify;

pub use types::*;
pub use verify::{
    canonical_tensor_execution_manifest_bytes, canonical_tensor_lineage_slice_bytes,
    canonicalized_tensor_execution_manifest, runtime_capability_binding_bundle_sha256,
    runtime_regime_binding_bundle_sha256, tensor_execution_manifest_sha256, tensor_lineage_slice,
    tensor_lineage_slice_sha256, tensor_state_node_sha256, unique_stored_payload_bytes,
    verify_tensor_execution_manifest, TensorExecutionManifestError,
    ValidatedTensorExecutionManifest,
};

#[cfg(test)]
mod tests;

#[cfg(test)]
mod adversarial_tests;
