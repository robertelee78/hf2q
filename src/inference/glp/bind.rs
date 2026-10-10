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

/// The post-layer residual site's steering state (ADR-053 dual hook sites).
///
/// `Dense` is one `[rows, hidden]` stream (Qwen). `MhcStreams` is the
/// DeepSeek-V4 hyper-connection fold — a `[rows, hc, hidden]` state whose
/// `residual_stream_post_layer` site is steered by the mHC kernel:
/// project-mode only (no add-mHC kernel exists) with directions of width
/// `hidden` (one shared direction steering every stream — what
/// `hf2q calibrate` exports) or `hc*hidden` (per-stream, the weightless
/// mHC discipline). Every other site, on every family, steers a dense
/// `[rows, hidden]` buffer and stays `hidden`-width.
#[derive(Clone, Copy, Debug)]
pub enum GlpResidualSite {
    Dense,
    MhcStreams { hc: u32 },
}

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
    /// `family_hooks` is the set of spec hook sites that family supports
    /// (ADR-053 dual hook sites: both the Qwen and DeepSeek engines pass
    /// `[residual_stream_post_layer, ffn_out_pre_residual]`). Per
    /// spec/GLP.md, a vector is calibrated for one site and "a reader whose
    /// hook does not match must refuse the file rather than apply it
    /// somewhere else" — the hooks are different tensors, not synonyms, so
    /// a vector whose `glp.hook_point` is outside the family's supported
    /// set is fatal here (naming the supported sites) instead of a silent
    /// reinterpretation at the apply point. `attn_out_pre_residual` is
    /// refused globally: spec-recognized, no hf2q family implements it.
    /// Binding a site does not by itself prove the family's forward graph
    /// applies it — an engine with no apply arm for a bound site fails its
    /// load by name rather than serving silently unsteered.
    ///
    /// Alpha precedence: request/CLI override > the file's
    /// `glp.alpha_default`.
    ///
    /// `residual_site` declares the state this family's
    /// `residual_stream_post_layer` site steers (ADR-053 dual hook
    /// sites). `Dense` accepts `hidden`-width directions only and puts
    /// no further mode constraint on any site; `MhcStreams` additionally
    /// accepts `hc*hidden` per-stream directions at that site and
    /// refuses `add` mode there with a named error — the mHC kernel
    /// implements the per-stream projection only, so refusing at bind
    /// beats failing (or silently no-oping) at runtime.
    pub fn bind_for_family(
        vector: GlpVector,
        alpha_override: Option<f32>,
        device: &MlxDevice,
        family_hooks: &[GlpHookPoint],
        model_num_layers: u32,
        model_hidden: u32,
        residual_site: GlpResidualSite,
    ) -> Result<Self, GlpError> {
        // S8: model compatibility is enforced BEFORE serving. A vector
        // naming layers the model does not have would be silently unused;
        // a direction whose width differs from the site's steering slice
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
        // ADR-053 dual hook sites: every site steers row slices of the
        // model's hidden width, so `hidden` is always accepted. The
        // DeepSeek post-layer residual site steers an mHC `[rows, hc,
        // hidden]` state and additionally accepts `hc*hidden` per-stream
        // directions (the weightless mHC discipline); writer-site vectors
        // stay `hidden`-only.
        let per_stream_width = match (residual_site, vector.hook_point) {
            (GlpResidualSite::MhcStreams { hc }, GlpHookPoint::ResidualStreamPostLayer) => {
                Some(hc as usize * model_hidden as usize)
            }
            _ => None,
        };
        if vector.width != model_hidden as usize && Some(vector.width) != per_stream_width {
            let accepted = match per_stream_width {
                Some(hc_hidden) => format!("hidden={model_hidden} or hc*hidden={hc_hidden}"),
                None => format!("hidden={model_hidden}"),
            };
            return Err(GlpError::Conformance(format!(
                "GLP direction width {} is not accepted at hook {} on this \
                 family (accepted widths: {accepted}); a mismatched width \
                 reads outside the direction buffer at apply time — refusing",
                vector.width,
                vector.hook_point.as_str()
            )));
        }
        // Global refusal first: spec-recognized, no hf2q family implements
        // this site — fatal even if a family set declared it, so the file
        // cannot be silently applied at another site.
        if vector.hook_point == GlpHookPoint::AttnOutPreResidual {
            return Err(GlpError::Conformance(
                "glp.hook_point attn_out_pre_residual is spec-recognized but \
                 no hf2q family implements that site; refusing"
                    .into(),
            ));
        }
        if !family_hooks.contains(&vector.hook_point) {
            let supported_sites = family_hooks
                .iter()
                .map(|hook| hook.as_str())
                .collect::<Vec<_>>()
                .join(", ");
            return Err(GlpError::Conformance(format!(
                "GLP vector declares glp.hook_point={} but this model family \
                 supports hook sites [{supported_sites}] (derived_at={}); \
                 refusing per spec — the hooks are different tensors and are \
                 not interchangeable",
                vector.hook_point.as_str(),
                vector.derived_at.as_deref().unwrap_or("<undeclared>")
            )));
        }
        // ADR-053 dual hook sites: the DeepSeek post-layer residual site
        // is project-mode-only — the mHC helpers implement the per-stream
        // projection kernel and no add-mHC kernel exists. Refuse the
        // (deepseek, residual, add) combination with a named error here
        // rather than failing (or silently no-oping) at runtime.
        if matches!(residual_site, GlpResidualSite::MhcStreams { .. })
            && vector.hook_point == GlpHookPoint::ResidualStreamPostLayer
            && vector.mode == GlpMode::Add
        {
            return Err(GlpError::Conformance(
                "glp.mode add is not implemented at hook \
                 residual_stream_post_layer on deepseek4: the post-layer mHC \
                 state is steered project-mode only (no add-mHC kernel \
                 exists); refusing at bind rather than failing at runtime"
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

    /// Family bind with the dense residual-site contract: the
    /// `residual_stream_post_layer` site steers a `[rows, hidden]` stream,
    /// so every site accepts `hidden`-width directions only and carries no
    /// extra mode constraint. The behavior every family had before the
    /// ADR-053 dual hook sites extension; families whose residual site
    /// steers mHC state (DeepSeek-V4) call [`Self::bind_for_family`].
    pub fn bind(
        vector: GlpVector,
        alpha_override: Option<f32>,
        device: &MlxDevice,
        family_hooks: &[GlpHookPoint],
        model_num_layers: u32,
        model_hidden: u32,
    ) -> Result<Self, GlpError> {
        Self::bind_for_family(
            vector,
            alpha_override,
            device,
            family_hooks,
            model_num_layers,
            model_hidden,
            GlpResidualSite::Dense,
        )
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
    /// than apply it somewhere else." A vector whose declared site is
    /// outside the family's supported site set (a residual-site vector
    /// handed to a writer-site-only family, or the reverse) is fatal, never
    /// a silent reinterpretation at the wrong tensor — the error names the
    /// family's supported sites.
    #[test]
    fn family_hook_outside_supported_set_is_fatal() {
        let device = MlxDevice::new().expect("MlxDevice");
        let residual_vector = test_vector(GlpMode::Project, GlpHookPoint::ResidualStreamPostLayer);
        let err = match BoundGlp::bind(
            residual_vector,
            None,
            &device,
            &[GlpHookPoint::FfnOutPreResidual],
            43,
            2,
        ) {
            Err(err) => err,
            Ok(_) => panic!("hook outside the family set must be fatal, bound anyway"),
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
                &[GlpHookPoint::ResidualStreamPostLayer],
                43,
                2
            ),
            Err(GlpError::Conformance(_))
        ));
    }

    /// The published GLP-29 structure: hook_point=ffn_out_pre_residual,
    /// derived_at=residual_stream_post_layer — the declared site transfer
    /// is legal and must bind on a family whose supported set contains the
    /// hook (the engine shape: both spec sites).
    #[test]
    fn declared_site_transfer_binds_on_matching_family() {
        let device = MlxDevice::new().expect("MlxDevice");
        let mut vector = test_vector(GlpMode::Project, GlpHookPoint::FfnOutPreResidual);
        vector.derived_at = Some("residual_stream_post_layer".into());
        let bound = BoundGlp::bind(
            vector,
            None,
            &device,
            &[GlpHookPoint::ResidualStreamPostLayer, GlpHookPoint::FfnOutPreResidual],
            43,
            2,
        )
        .expect("hook inside the family set must bind");
        assert_eq!(bound.alpha, 1.0);
        assert!(bound.direction_for(3).is_some());
    }

    /// attn_out_pre_residual is spec-recognized but no hf2q family
    /// implements it; binding one is fatal even when the family set
    /// declares it, so the file cannot be silently applied at another site.
    #[test]
    fn unimplemented_hook_site_is_fatal_even_when_declared() {
        let device = MlxDevice::new().expect("MlxDevice");
        let vector = test_vector(GlpMode::Project, GlpHookPoint::AttnOutPreResidual);
        assert!(matches!(
            BoundGlp::bind(
                vector,
                None,
                &device,
                &[GlpHookPoint::AttnOutPreResidual],
                43,
                2
            ),
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
        let bound = BoundGlp::bind(
            project,
            None,
            &device,
            &[GlpHookPoint::FfnOutPreResidual],
            43,
            2,
        )
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
        let bound = BoundGlp::bind(
            add,
            None,
            &device,
            &[GlpHookPoint::ResidualStreamPostLayer],
            43,
            2,
        )
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
