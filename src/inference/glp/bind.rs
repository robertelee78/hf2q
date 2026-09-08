//! GPU-side GLP binding: upload a conformance-checked vector's directions to
//! the device once at serve start. Family-agnostic; the family forward code
//! decides which activation buffer and which direction.
//!
//! The apply arithmetic contract (CPU reference) lives in `apply.rs`. MLX
//! dispatch for the steered hook is wired per family.

use mlx_native::metal;
use mlx_native::{DType, MlxBuffer, MlxDevice};

use super::reader::{GlpMode, GlpVector};
use super::GlpError;

/// A GLP vector bound to a device (directions uploaded once at serve start).
pub struct BoundGlp {
    pub vector: GlpVector,
    pub alpha: f32,
    /// layer N → device-resident direction buffer (fp32, [width])
    pub device_directions: std::collections::BTreeMap<u32, MlxBuffer>,
}

impl BoundGlp {
    /// Bind a loaded vector to the device. Alpha precedence:
    /// request/CLI override > the file's `glp.alpha_default`.
    pub fn bind(
        vector: GlpVector,
        alpha_override: Option<f32>,
        device: &MlxDevice,
    ) -> Result<Self, GlpError> {
        let alpha = alpha_override.unwrap_or(vector.alpha_default);
        if !alpha.is_finite() || alpha < 0.0 {
            return Err(GlpError::Conformance(format!(
                "glp alpha {alpha} must be a finite non-negative number"
            )));
        }
        let mut device_directions = std::collections::BTreeMap::new();
        for (layer, direction) in &vector.layers {
            // Normalize the direction to unit norm at bind time. The kernel
            // then divides the per-row dot by the (unit) norm squared = 1,
            // and alpha is the sole dose control. This is the project-mode
            // contract: `h -= alpha * (h·d̂)d̂` with d̂ = d/‖d‖.
            //
            // The norm computation MUST be in f64: published GLP vectors
            // (e.g. GLP-29) store raw mean-difference directions with norms
            // up to ~1e29, whose squared norm overflows f32 (~3.4e38).
            let norm_sq_f64: f64 = direction.iter().map(|x| (*x as f64) * (*x as f64)).sum();
            let norm_f64 = norm_sq_f64.sqrt();
            if norm_f64 <= 0.0 {
                return Err(GlpError::Conformance(format!(
                    "direction.{layer} has zero norm"
                )));
            }
            let normalized: Vec<f32> = direction
                .iter()
                .map(|x| (*x as f64 / norm_f64) as f32)
                .collect();
            let byte_len = std::mem::size_of_val(normalized.as_slice());
            let raw = device.metal_device().new_buffer_with_data(
                normalized.as_ptr().cast(),
                byte_len as u64,
                metal::MTLResourceOptions::StorageModeShared,
            );
            let buffer = MlxBuffer::from_raw(raw, DType::F32, vec![normalized.len()]);
            device_directions.insert(*layer, buffer);
        }
        Ok(Self { vector, alpha, device_directions })
    }

    pub fn mode(&self) -> GlpMode {
        self.vector.mode
    }

    pub fn direction_for(&self, layer: u32) -> Option<&MlxBuffer> {
        self.device_directions.get(&layer)
    }
}

#[cfg(test)]
mod tests {
    // Device-binding tests need Metal; they live behind the hardware gate.
    // The pure-Rust arithmetic contract is covered by apply.rs tests.
}
