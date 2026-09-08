//! The apply arithmetic for GLP steering (spec/GLP.md).
//!
//! Both operations act on one layer-local activation tensor `h` (any shape —
//! the family hook flattens/reshapes as needed and calls with the width
//! matching the vector):
//!
//! - `project`: `h ← h − α(h·d̂)d̂` — remove the component along d̂.
//!   alpha is NEVER folded into the vector (projection is quadratic in ‖d‖).
//! - `add`: `h ← h + α·v` — llama.cpp control-vector convention.

use super::reader::GlpError;

/// Project `h` onto the orthogonal complement of `direction`, scaled by alpha.
/// `direction` need not be unit-norm; it is normalized in-place cost-free.
pub fn apply_layer_project(
    h: &mut [f32],
    direction: &[f32],
    alpha: f32,
) -> Result<(), GlpError> {
    if h.len() != direction.len() {
        return Err(GlpError::Conformance(format!(
            "activation width {} != direction width {}",
            h.len(),
            direction.len()
        )));
    }
    let norm_sq: f32 = direction.iter().map(|x| x * x).sum();
    if norm_sq <= f32::EPSILON {
        return Err(GlpError::Conformance(
            "project direction is near-zero; cannot normalize".into(),
        ));
    }
    let dot: f32 = h.iter().zip(direction.iter()).map(|(a, b)| a * b).sum();
    let scale = alpha * dot / norm_sq;
    for (hv, dv) in h.iter_mut().zip(direction.iter()) {
        *hv -= scale * dv;
    }
    Ok(())
}

/// Add `alpha * direction` to `h` (additive control-vector convention).
pub fn apply_layer_add(h: &mut [f32], direction: &[f32], alpha: f32) -> Result<(), GlpError> {
    if h.len() != direction.len() {
        return Err(GlpError::Conformance(format!(
            "activation width {} != direction width {}",
            h.len(),
            direction.len()
        )));
    }
    for (hv, dv) in h.iter_mut().zip(direction.iter()) {
        *hv += alpha * dv;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn project_removes_the_component_along_d() {
        // h = 3·d̂ + rest; after projection with alpha=1 the d-component is gone
        let d = vec![1.0, 0.0, 0.0];
        let mut h = vec![3.0, 1.0, -2.0];
        apply_layer_project(&mut h, &d, 1.0).unwrap();
        assert!((h[0]).abs() < 1e-6);
        assert!((h[1] - 1.0).abs() < 1e-6);
        assert!((h[2] + 2.0).abs() < 1e-6);
    }

    #[test]
    fn project_alpha_two_overremoves_by_design() {
        let d = vec![0.0, 2.0, 0.0]; // non-unit: norm^2 = 4
        let mut h = vec![0.0, 4.0, 1.0];
        apply_layer_project(&mut h, &d, 2.0).unwrap();
        // dot = 8, scale = 2*8/4 = 4 → h[1] = 4 - 4*2 = -4
        assert!((h[1] + 4.0).abs() < 1e-5);
        assert!((h[2] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn add_scales_direction_by_alpha() {
        let d = vec![1.0, 2.0, 3.0];
        let mut h = vec![0.0, 0.0, 0.0];
        apply_layer_add(&mut h, &d, 0.5).unwrap();
        assert!((h[0] - 0.5).abs() < 1e-6);
        assert!((h[1] - 1.0).abs() < 1e-6);
        assert!((h[2] - 1.5).abs() < 1e-6);
    }

    #[test]
    fn width_mismatch_is_an_error_not_a_panic() {
        let d = vec![1.0];
        let mut h = vec![1.0, 2.0];
        assert!(apply_layer_project(&mut h, &d, 1.0).is_err());
        assert!(apply_layer_add(&mut h, &d, 1.0).is_err());
    }

    #[test]
    fn zero_direction_is_rejected_for_project() {
        let d = vec![0.0, 0.0];
        let mut h = vec![1.0, 2.0];
        assert!(apply_layer_project(&mut h, &d, 1.0).is_err());
    }
}
