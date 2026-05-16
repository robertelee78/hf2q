//! Calibration module — pure-Rust implementations of weight-quantization
//! calibrators used by hf2q's IR-quantize and native-quantize paths.
//!
//! ## ADR-014 P6 — Imatrix calibrator (this module's first inhabitant)
//!
//! [`imatrix`] is the pure-Rust port of llama.cpp's importance-matrix
//! algorithm (`/opt/llama.cpp/tools/imatrix/imatrix.cpp`). At a forward
//! pass through a calibration corpus, the calibrator captures the
//! input activation vector at every Linear layer and accumulates
//! `x[col]² · 1.0` per column. The mean of squared activations becomes
//! the **per-column importance weight** consumed by the per-column-
//! weighted MSE inside the k-quant codebook search at quantize time
//! (ADR-014 Decision 11, lands in P7 alongside the
//! [`Calibrator`]-trait orthogonal split).
//!
//! ## ADR-014 P7 — orthogonal `Calibrator × OutputFormat` split
//!
//! P7 added the [`calibrator::Calibrator`] trait + [`calibrator::NoneCalibrator`]
//! impl, the [`imatrix_calibrator::ImatrixCalibrator`] and
//! [`dwq_calibrator::DwqCalibrator`] trait impls, and migrated `dwq.rs`
//! / `dwq_activation.rs` / `sensitivity.rs` / `apex.rs` from
//! `src/quantize/` into this module (Layout A, iter-8). The imatrix
//! algorithm + Stats + Collector stays in [`imatrix`]; the trait-driven
//! calibrator wrapper lives in [`imatrix_calibrator`]. The DWQ
//! orchestration moved alongside its sensitivity helpers — [`dwq`],
//! [`dwq_activation`], and [`sensitivity`] are now siblings, with
//! [`dwq_calibrator`] as the trait wrapper.

pub mod apex;
pub mod autograd;
pub mod autograd_gpu;
pub mod autograd_gpu_tape;
pub mod cache;
pub mod calibrator;
pub mod dwq;
pub mod dwq_activation;
pub mod dwq_calibrator;
pub mod dynamic_quant;
pub mod dynamic_quant_gpu;
pub mod qdq_gpu;
pub mod qwen35_attention_block;
pub mod qwen35_ffn;
pub mod qwen35_gated_delta;
pub mod qwen35_layer;
pub mod qwen35_moe;
pub mod qwen35_model;
pub mod qwen35_gguf_adapter;
pub mod calibration_batcher;
pub mod sensitivity_comparison;
pub mod adam;
pub mod dwq_benchmark;
pub mod dwq_loop;
pub mod dwq_targets;
pub mod fd_sensitivity;
pub mod gguf_teacher;
pub mod hf_safetensors_teacher;
pub mod dwq_e2e;
pub mod imatrix;
pub mod imatrix_calibrator;
pub mod imatrix_xvalidate;
pub mod sensitivity;

#[cfg(test)]
pub(crate) mod test_support {
    /// Serialises tests that mutate process-wide cache environment variables.
    pub(crate) static CACHE_ENV_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
}
