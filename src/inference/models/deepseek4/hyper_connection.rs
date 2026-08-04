//! CPU reference for DeepSeek-V4 Hyper-Connection splitting and Sinkhorn mix.

use thiserror::Error;

#[derive(Clone, Debug, PartialEq)]
pub struct HyperConnectionMix {
    pub pre: Vec<f32>,
    pub post: Vec<f32>,
    /// Row-major `hc_mult × hc_mult` doubly-normalized residual mix.
    pub combination: Vec<f32>,
}

#[derive(Debug, Error, Eq, PartialEq)]
pub enum HyperConnectionError {
    #[error("hc_mult must be greater than zero")]
    Empty,
    #[error("sinkhorn_iters must be greater than zero")]
    NoIterations,
    #[error("expected {expected} mix values, got {actual}")]
    MixCount { expected: usize, actual: usize },
    #[error("expected {expected} base values, got {actual}")]
    BaseCount { expected: usize, actual: usize },
    #[error("Hyper-Connection input contains a non-finite value")]
    NonFinite,
}

fn sigmoid(value: f32) -> f32 {
    if value >= 0.0 {
        1.0 / (1.0 + (-value).exp())
    } else {
        let exp = value.exp();
        exp / (1.0 + exp)
    }
}

fn normalize_columns(matrix: &mut [f32], size: usize, eps: f32) {
    for column in 0..size {
        let sum: f32 = (0..size).map(|row| matrix[row * size + column]).sum();
        for row in 0..size {
            matrix[row * size + column] /= sum + eps;
        }
    }
}

fn normalize_rows(matrix: &mut [f32], size: usize, eps: f32) {
    for row in 0..size {
        let offset = row * size;
        let sum: f32 = matrix[offset..offset + size].iter().sum();
        for value in &mut matrix[offset..offset + size] {
            *value /= sum + eps;
        }
    }
}

/// Split `(2 + hc_mult) * hc_mult` learned mix logits into the pre, post,
/// and Sinkhorn-normalized residual combination used by one transformer sublayer.
pub fn split_sinkhorn(
    mixes: &[f32],
    scale: [f32; 3],
    base: &[f32],
    hc_mult: usize,
    sinkhorn_iters: usize,
    eps: f32,
) -> Result<HyperConnectionMix, HyperConnectionError> {
    if hc_mult == 0 {
        return Err(HyperConnectionError::Empty);
    }
    if sinkhorn_iters == 0 {
        return Err(HyperConnectionError::NoIterations);
    }
    let expected = (2 + hc_mult) * hc_mult;
    if mixes.len() != expected {
        return Err(HyperConnectionError::MixCount {
            expected,
            actual: mixes.len(),
        });
    }
    if base.len() != expected {
        return Err(HyperConnectionError::BaseCount {
            expected,
            actual: base.len(),
        });
    }
    if !eps.is_finite()
        || eps < 0.0
        || scale.iter().any(|value| !value.is_finite())
        || mixes.iter().any(|value| !value.is_finite())
        || base.iter().any(|value| !value.is_finite())
    {
        return Err(HyperConnectionError::NonFinite);
    }

    let pre = (0..hc_mult)
        .map(|index| sigmoid(mixes[index] * scale[0] + base[index]) + eps)
        .collect();
    let post = (0..hc_mult)
        .map(|index| {
            let source = hc_mult + index;
            2.0 * sigmoid(mixes[source] * scale[1] + base[source])
        })
        .collect();

    let combination_offset = 2 * hc_mult;
    let mut combination = vec![0.0; hc_mult * hc_mult];
    for row in 0..hc_mult {
        let row_offset = row * hc_mult;
        let row_max = (0..hc_mult)
            .map(|column| {
                let source = combination_offset + row_offset + column;
                mixes[source] * scale[2] + base[source]
            })
            .fold(f32::NEG_INFINITY, f32::max);
        let mut row_sum = 0.0;
        for column in 0..hc_mult {
            let source = combination_offset + row_offset + column;
            let value = (mixes[source] * scale[2] + base[source] - row_max).exp();
            combination[row_offset + column] = value;
            row_sum += value;
        }
        for value in &mut combination[row_offset..row_offset + hc_mult] {
            *value = *value / row_sum + eps;
        }
    }
    normalize_columns(&mut combination, hc_mult, eps);
    for _ in 1..sinkhorn_iters {
        normalize_rows(&mut combination, hc_mult, eps);
        normalize_columns(&mut combination, hc_mult, eps);
    }

    Ok(HyperConnectionMix {
        pre,
        post,
        combination,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zero_logits_have_official_symmetric_solution() {
        let hc = 4;
        let values = vec![0.0; (2 + hc) * hc];
        let result = split_sinkhorn(&values, [1.0; 3], &values, hc, 20, 1e-6).unwrap();
        for value in result.pre {
            assert!((value - 0.500001).abs() < 1e-6);
        }
        for value in result.post {
            assert!((value - 1.0).abs() < 1e-6);
        }
        for value in result.combination {
            assert!((value - 0.25).abs() < 2e-6);
        }
    }

    #[test]
    fn sinkhorn_output_is_nearly_doubly_stochastic() {
        let hc = 4;
        let mut values = vec![0.0; (2 + hc) * hc];
        for (index, value) in values.iter_mut().enumerate() {
            *value = (index as f32 - 11.5) * 0.17;
        }
        let base: Vec<f32> = values.iter().map(|value| -*value * 0.25).collect();
        let result = split_sinkhorn(&values, [0.7, 1.3, 0.9], &base, hc, 20, 1e-6).unwrap();
        for row in 0..hc {
            let sum: f32 = result.combination[row * hc..(row + 1) * hc].iter().sum();
            assert!((sum - 1.0).abs() < 2e-5, "row {row} sum={sum}");
        }
        for column in 0..hc {
            let sum: f32 = (0..hc)
                .map(|row| result.combination[row * hc + column])
                .sum();
            assert!((sum - 1.0).abs() < 2e-5, "column {column} sum={sum}");
        }
    }

    #[test]
    fn invalid_shapes_and_iterations_fail_closed() {
        assert_eq!(
            split_sinkhorn(&[], [1.0; 3], &[], 0, 20, 1e-6).unwrap_err(),
            HyperConnectionError::Empty
        );
        assert_eq!(
            split_sinkhorn(&[0.0; 24], [1.0; 3], &[0.0; 24], 4, 0, 1e-6).unwrap_err(),
            HyperConnectionError::NoIterations
        );
        assert_eq!(
            split_sinkhorn(&[0.0; 23], [1.0; 3], &[0.0; 24], 4, 20, 1e-6).unwrap_err(),
            HyperConnectionError::MixCount {
                expected: 24,
                actual: 23
            }
        );
    }
}
