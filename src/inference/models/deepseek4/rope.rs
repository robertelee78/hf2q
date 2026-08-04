//! DeepSeek-V4 rotary embedding and YaRN frequency reference.

use std::f32::consts::PI;

use thiserror::Error;

#[derive(Debug, Error, Eq, PartialEq)]
pub enum RopeError {
    #[error("rotary dimension must be positive and even, got {dim}")]
    InvalidDimension { dim: usize },
    #[error("rope base and factor must be finite and positive")]
    InvalidScale,
    #[error("input length {actual} is not divisible by rotary dimension {dim}")]
    InputShape { dim: usize, actual: usize },
    #[error("expected {expected} positions, got {actual}")]
    PositionCount { expected: usize, actual: usize },
    #[error("rotary input contains a non-finite value")]
    NonFinite,
}

fn correction_dimension(rotations: f32, dim: usize, base: f32, original_context: usize) -> f32 {
    dim as f32 * (original_context as f32 / (rotations * 2.0 * PI)).ln() / (2.0 * base.ln())
}

/// Per-pair angular frequencies used by the official 0731 implementation.
pub fn yarn_frequencies(
    dim: usize,
    original_context: usize,
    base: f32,
    factor: f32,
    beta_fast: f32,
    beta_slow: f32,
) -> Result<Vec<f32>, RopeError> {
    if dim == 0 || dim % 2 != 0 {
        return Err(RopeError::InvalidDimension { dim });
    }
    if !base.is_finite() || !factor.is_finite() || base <= 0.0 || factor <= 0.0 {
        return Err(RopeError::InvalidScale);
    }
    if !beta_fast.is_finite() || !beta_slow.is_finite() || beta_fast <= 0.0 || beta_slow <= 0.0 {
        return Err(RopeError::InvalidScale);
    }

    let pairs = dim / 2;
    let mut frequencies: Vec<f32> = (0..pairs)
        .map(|pair| 1.0 / base.powf((2 * pair) as f32 / dim as f32))
        .collect();
    if original_context == 0 {
        return Ok(frequencies);
    }

    let low = correction_dimension(beta_fast, dim, base, original_context)
        .floor()
        .max(0.0) as usize;
    let high = correction_dimension(beta_slow, dim, base, original_context)
        .ceil()
        .min((dim - 1) as f32) as usize;
    let upper = if low == high {
        high as f32 + 0.001
    } else {
        high as f32
    };
    for (pair, frequency) in frequencies.iter_mut().enumerate() {
        let ramp = ((pair as f32 - low as f32) / (upper - low as f32)).clamp(0.0, 1.0);
        let smooth = 1.0 - ramp;
        *frequency = *frequency / factor * (1.0 - smooth) + *frequency * smooth;
    }
    Ok(frequencies)
}

/// Apply RoPE to consecutive rows in place. Each row contains exactly `dim`
/// rotary features and gets the corresponding absolute position. `inverse`
/// conjugates the phase and is used after sparse attention.
pub fn apply_rotary(
    values: &mut [f32],
    positions: &[usize],
    frequencies: &[f32],
    inverse: bool,
) -> Result<(), RopeError> {
    let dim = frequencies.len() * 2;
    if dim == 0 {
        return Err(RopeError::InvalidDimension { dim });
    }
    if values.len() % dim != 0 {
        return Err(RopeError::InputShape {
            dim,
            actual: values.len(),
        });
    }
    let rows = values.len() / dim;
    if positions.len() != rows {
        return Err(RopeError::PositionCount {
            expected: rows,
            actual: positions.len(),
        });
    }
    if values.iter().any(|value| !value.is_finite())
        || frequencies.iter().any(|value| !value.is_finite())
    {
        return Err(RopeError::NonFinite);
    }

    let direction = if inverse { -1.0 } else { 1.0 };
    for (row, &position) in positions.iter().enumerate() {
        let offset = row * dim;
        for (pair, frequency) in frequencies.iter().enumerate() {
            let angle = direction * position as f32 * frequency;
            let (sin, cos) = angle.sin_cos();
            let real = values[offset + 2 * pair];
            let imaginary = values[offset + 2 * pair + 1];
            values[offset + 2 * pair] = real * cos - imaginary * sin;
            values[offset + 2 * pair + 1] = real * sin + imaginary * cos;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn base_rope_has_expected_geometric_frequencies() {
        let frequencies = yarn_frequencies(8, 0, 10_000.0, 16.0, 32.0, 1.0).unwrap();
        let expected = [1.0, 0.1, 0.01, 0.001];
        for (actual, expected) in frequencies.iter().zip(expected) {
            assert!((actual - expected).abs() < 1e-7);
        }
    }

    #[test]
    fn yarn_preserves_fast_pairs_and_interpolates_slow_pairs() {
        let base = yarn_frequencies(64, 0, 10_000.0, 16.0, 32.0, 1.0).unwrap();
        let yarn = yarn_frequencies(64, 65_536, 10_000.0, 16.0, 32.0, 1.0).unwrap();
        assert_eq!(base[0], yarn[0]);
        assert!(yarn.last().unwrap() < base.last().unwrap());
        assert!(yarn.last().unwrap() > &(base.last().unwrap() / 16.0));
        assert!(yarn[16] <= base[16]);
        assert!(yarn[16] >= base[16] / 16.0);
    }

    #[test]
    fn inverse_rotation_recovers_input() {
        let frequencies = yarn_frequencies(8, 65_536, 10_000.0, 16.0, 32.0, 1.0).unwrap();
        let original: Vec<f32> = (0..24).map(|value| value as f32 * 0.125 - 1.0).collect();
        let mut rotated = original.clone();
        apply_rotary(&mut rotated, &[0, 17, 1_000_000], &frequencies, false).unwrap();
        apply_rotary(&mut rotated, &[0, 17, 1_000_000], &frequencies, true).unwrap();
        for (actual, expected) in rotated.iter().zip(original) {
            assert!((actual - expected).abs() < 2e-5, "{actual} != {expected}");
        }
    }

    #[test]
    fn malformed_shapes_fail_closed() {
        assert_eq!(
            yarn_frequencies(7, 0, 10_000.0, 1.0, 32.0, 1.0).unwrap_err(),
            RopeError::InvalidDimension { dim: 7 }
        );
        assert_eq!(
            apply_rotary(&mut [0.0; 3], &[0], &[1.0, 0.1], false).unwrap_err(),
            RopeError::InputShape { dim: 4, actual: 3 }
        );
    }
}
