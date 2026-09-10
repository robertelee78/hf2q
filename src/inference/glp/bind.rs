//! GPU-side GLP binding: upload a conformance-checked vector's directions to
//! the device once at serve start. Family-agnostic; the family forward code
//! decides which activation buffer and which direction.
//!
//! The apply arithmetic contract (CPU reference) lives in `apply.rs`. MLX
//! dispatch for the steered hook is wired per family.

use mlx_native::metal;
use mlx_native::{DType, MlxBuffer, MlxDevice};

use super::reader::{GlpHookPoint, GlpMode, GlpVector};
use super::GlpError;

/// A GLP vector bound to a device (directions uploaded once at serve start).
pub struct BoundGlp {
    pub vector: GlpVector,
    pub alpha: f32,
    /// layer N → device-resident direction buffer (fp32, [width])
    pub device_directions: std::collections::BTreeMap<u32, MlxBuffer>,
}

impl BoundGlp {
    /// Bind a loaded vector to the device for one model family.
    ///
    /// `family_hook` is the site that family's forward graph actually
    /// steers (DeepSeek-V4: the FFN writer pre-fold; Qwen: the post-layer
    /// residual). Per spec/GLP.md, a vector is calibrated for one site and
    /// "a reader whose hook does not match must refuse the file rather than
    /// apply it somewhere else" — the hooks are different tensors, not
    /// synonyms, so a mismatch is fatal here instead of a silent
    /// reinterpretation at the apply point.
    ///
    /// Alpha precedence: request/CLI override > the file's
    /// `glp.alpha_default`.
    pub fn bind(
        vector: GlpVector,
        alpha_override: Option<f32>,
        device: &MlxDevice,
        family_hook: GlpHookPoint,
        model_num_layers: u32,
        model_hidden: u32,
    ) -> Result<Self, GlpError> {
        // S8: model compatibility is enforced BEFORE serving. A vector
        // naming layers the model does not have would be silently unused;
        // a direction whose width differs from the model's hidden size
        // would read outside its buffer at apply time (the Qwen dispatcher
        // had no width check). Loading a file successfully is not proof
        // that its intervention executes.
        for layer in vector.layers.keys() {
            if *layer >= model_num_layers {
                return Err(GlpError::Conformance(format!(
                    "direction.{layer} names a layer this model does not have \
                     (num_layers={model_num_layers}); the direction would be \
                     silently unused — refusing"
                )));
            }
        }
        if vector.width != model_hidden as usize {
            return Err(GlpError::Conformance(format!(
                "GLP direction width {} != model hidden size {model_hidden}; \
                 a mismatched width reads outside the direction buffer at \
                 apply time — refusing",
                vector.width
            )));
        }
        if vector.hook_point != family_hook {
            return Err(GlpError::Conformance(format!(
                "GLP vector declares glp.hook_point={} but this model family \
                 applies at {} (derived_at={}); refusing per spec — the hooks \
                 are different tensors and are not interchangeable",
                vector.hook_point.as_str(),
                family_hook.as_str(),
                vector.derived_at.as_deref().unwrap_or("<undeclared>")
            )));
        }
        if family_hook == GlpHookPoint::AttnOutPreResidual {
            return Err(GlpError::Conformance(
                "glp.hook_point attn_out_pre_residual is spec-recognized but \
                 no hf2q family implements that site; refusing"
                    .into(),
            ));
        }
        let alpha = alpha_override.unwrap_or(vector.alpha_default);
        if !alpha.is_finite() || alpha < 0.0 {
            return Err(GlpError::Conformance(format!(
                "glp alpha {alpha} must be a finite non-negative number"
            )));
        }
        let mut device_directions = std::collections::BTreeMap::new();
        for (layer, direction) in &vector.layers {
            // Non-finite elements are fatal in BOTH modes before any GPU
            // upload: a NaN makes every downstream comparison false (the
            // `norm <= 0` guard cannot catch it — NaN comparisons are
            // false), and normalizing an infinite value produces NaNs that
            // silently contaminate the model computation.
            if direction.iter().any(|x| !x.is_finite()) {
                return Err(GlpError::Conformance(format!(
                    "direction.{layer} contains non-finite values (NaN or \
                     infinity); refusing before device upload"
                )));
            }
            // Project mode: normalize the direction to unit norm at bind
            // time. The kernel then divides the per-row dot by the (unit)
            // norm squared = 1, and alpha is the sole dose control. This is
            // the project-mode contract: `h -= alpha * (h·d̂)d̂` with
            // d̂ = d/‖d‖.
            //
            // Add mode: upload the RAW direction. The additive path folds
            // strength into the data (spec) — normalizing would destroy the
            // baked-in scale of a legacy control vector; alpha (default 1.0
            // for files without glp.alpha_default) scales it at apply time.
            //
            // The norm computation MUST be in f64: published GLP vectors
            // (e.g. GLP-29) store raw mean-difference directions with norms
            // up to ~1e29, whose squared norm overflows f32 (~3.4e38).
            let normalized: Vec<f32> = match vector.mode {
                GlpMode::Project => {
                    let norm_sq_f64: f64 =
                        direction.iter().map(|x| (*x as f64) * (*x as f64)).sum();
                    let norm_f64 = norm_sq_f64.sqrt();
                    if !norm_f64.is_finite() || norm_f64 <= 0.0 {
                        return Err(GlpError::Conformance(format!(
                            "direction.{layer} has an invalid norm ({norm_f64}); \
                             refusing before device upload"
                        )));
                    }
                    direction
                        .iter()
                        .map(|x| (*x as f64 / norm_f64) as f32)
                        .collect()
                }
                GlpMode::Add => direction.clone(),
            };
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
    use super::*;
    use std::collections::BTreeMap;

    fn test_vector(mode: GlpMode, hook_point: GlpHookPoint) -> GlpVector {
        GlpVector {
            mode,
            hook_point,
            derived_at: None,
            alpha_default: 1.0,
            rank: 1,
            layers: BTreeMap::from([(
                3u32,
                vec![3.0f32, 4.0], // norm 5 — chosen so normalization is observable
            )]),
            width: 2,
            content_sha256: None,
            method: None,
            base_model_name: None,
        }
    }

    /// Spec: "a reader whose hook does not match must refuse the file rather
    /// than apply it somewhere else." A residual-site vector handed to the
    /// DeepSeek FFN-writer bind (or the reverse) is fatal, never a silent
    /// reinterpretation at the wrong tensor.
    #[test]
    fn family_hook_mismatch_is_fatal() {
        let device = MlxDevice::new().expect("MlxDevice");
        let residual_vector = test_vector(GlpMode::Project, GlpHookPoint::ResidualStreamPostLayer);
        let err = match BoundGlp::bind(
            residual_vector,
            None,
            &device,
            GlpHookPoint::FfnOutPreResidual,
            43,
            2,
        ) {
            Err(err) => err,
            Ok(_) => panic!("hook mismatch must be fatal, bound anyway"),
        };
        match err {
            GlpError::Conformance(message) => {
                assert!(message.contains("residual_stream_post_layer"), "{message}");
                assert!(message.contains("ffn_out_pre_residual"), "{message}");
            }
            other => panic!("expected conformance failure, got {other:?}"),
        }

        let writer_vector = test_vector(GlpMode::Project, GlpHookPoint::FfnOutPreResidual);
        assert!(matches!(
            BoundGlp::bind(
                writer_vector,
                None,
                &device,
                GlpHookPoint::ResidualStreamPostLayer,
                43,
                2
            ),
            Err(GlpError::Conformance(_))
        ));
    }

    /// The published GLP-29 structure: hook_point=ffn_out_pre_residual,
    /// derived_at=residual_stream_post_layer — the declared site transfer
    /// is legal and must bind on the matching family.
    #[test]
    fn declared_site_transfer_binds_on_matching_family() {
        let device = MlxDevice::new().expect("MlxDevice");
        let mut vector = test_vector(GlpMode::Project, GlpHookPoint::FfnOutPreResidual);
        vector.derived_at = Some("residual_stream_post_layer".into());
        let bound = BoundGlp::bind(vector, None, &device, GlpHookPoint::FfnOutPreResidual, 43, 2)
            .expect("matching hook must bind");
        assert_eq!(bound.alpha, 1.0);
        assert!(bound.direction_for(3).is_some());
    }

    /// attn_out_pre_residual is spec-recognized but no hf2q family
    /// implements it; binding one is fatal even when the declaration
    /// matches, so the file cannot be silently applied at another site.
    #[test]
    fn unimplemented_hook_site_is_fatal_even_when_declared() {
        let device = MlxDevice::new().expect("MlxDevice");
        let vector = test_vector(GlpMode::Project, GlpHookPoint::AttnOutPreResidual);
        assert!(matches!(
            BoundGlp::bind(vector, None, &device, GlpHookPoint::AttnOutPreResidual, 43, 2),
            Err(GlpError::Conformance(_))
        ));
    }

    /// Project mode normalizes at bind (alpha is the sole dose); add mode
    /// uploads the RAW direction (the additive path folds strength into
    /// the data — normalizing would destroy a legacy control vector's
    /// baked-in scale).
    #[test]
    fn mode_decides_normalization_at_bind() {
        let device = MlxDevice::new().expect("MlxDevice");

        let project = test_vector(GlpMode::Project, GlpHookPoint::FfnOutPreResidual);
        let bound = BoundGlp::bind(project, None, &device, GlpHookPoint::FfnOutPreResidual, 43, 2)
            .expect("bind project");
        let normalized: Vec<f32> = bound
            .direction_for(3)
            .unwrap()
            .as_slice::<f32>()
            .unwrap()
            .to_vec();
        let norm: f64 = normalized.iter().map(|x| (*x as f64) * (*x as f64)).sum::<f64>().sqrt();
        assert!((norm - 1.0).abs() < 1e-4, "project direction must be unit-norm, got {norm}");

        let add = test_vector(GlpMode::Add, GlpHookPoint::ResidualStreamPostLayer);
        let bound = BoundGlp::bind(add, None, &device, GlpHookPoint::ResidualStreamPostLayer, 43, 2)
            .expect("bind add");
        let raw: Vec<f32> = bound
            .direction_for(3)
            .unwrap()
            .as_slice::<f32>()
            .unwrap()
            .to_vec();
        assert_eq!(raw, vec![3.0, 4.0], "add direction must upload raw");
    }
}
