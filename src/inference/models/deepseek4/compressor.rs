//! CPU references for DeepSeek-V4 learned KV compression and index selection.
//!
//! The production Metal path consumes already-projected KV and gate tensors.
//! These routines deliberately start at that boundary so kernel parity tests do
//! not depend on a second matrix-multiplication implementation.

use thiserror::Error;

#[derive(Debug, Error, Eq, PartialEq)]
pub enum CompressorError {
    #[error("compression ratio and feature dimension must be greater than zero")]
    EmptyShape,
    #[error("expected {expected} projected values, got {actual}")]
    ProjectionShape { expected: usize, actual: usize },
    #[error("expected {expected} APE values, got {actual}")]
    ApeShape { expected: usize, actual: usize },
    #[error("expected {expected} query values, got {actual}")]
    QueryShape { expected: usize, actual: usize },
    #[error("expected {expected} indexer weights, got {actual}")]
    WeightShape { expected: usize, actual: usize },
    #[error("compressed KV length {actual} is not divisible by dimension {dim}")]
    KvShape { dim: usize, actual: usize },
    #[error("compression/indexer input contains a non-finite value")]
    NonFinite,
}

fn softmax_weighted_sum(
    values: &[f32],
    scores: &[f32],
    rows: usize,
    dim: usize,
    output: &mut [f32],
) {
    for column in 0..dim {
        let max = (0..rows)
            .map(|row| scores[row * dim + column])
            .fold(f32::NEG_INFINITY, f32::max);
        let mut denominator = 0.0;
        let mut numerator = 0.0;
        for row in 0..rows {
            let score = scores[row * dim + column];
            let weight = if score.is_infinite() && score.is_sign_negative() {
                0.0
            } else {
                (score - max).exp()
            };
            denominator += weight;
            numerator += values[row * dim + column] * weight;
        }
        output[column] = numerator / denominator;
    }
}

/// Compress every complete group in a prefill from already-projected KV and
/// gate tensors. `overlap` is the official ratio-4 layout: the first half of
/// each projection is shifted forward one group and pooled together with the
/// current group's second half. Incomplete trailing tokens are not emitted.
pub fn compress_prefill_projected(
    kv: &[f32],
    scores: &[f32],
    ape: &[f32],
    tokens: usize,
    ratio: usize,
    dim: usize,
    overlap: bool,
) -> Result<Vec<f32>, CompressorError> {
    if ratio == 0 || dim == 0 {
        return Err(CompressorError::EmptyShape);
    }
    let halves = if overlap { 2 } else { 1 };
    let projected_dim = halves * dim;
    let expected = tokens.saturating_mul(projected_dim);
    if kv.len() != expected {
        return Err(CompressorError::ProjectionShape {
            expected,
            actual: kv.len(),
        });
    }
    if scores.len() != expected {
        return Err(CompressorError::ProjectionShape {
            expected,
            actual: scores.len(),
        });
    }
    let expected_ape = ratio.saturating_mul(projected_dim);
    if ape.len() != expected_ape {
        return Err(CompressorError::ApeShape {
            expected: expected_ape,
            actual: ape.len(),
        });
    }
    if kv
        .iter()
        .chain(scores)
        .chain(ape)
        .any(|value| !value.is_finite())
    {
        return Err(CompressorError::NonFinite);
    }

    let groups = tokens / ratio;
    let mut output = vec![0.0; groups * dim];
    let rows = if overlap { 2 * ratio } else { ratio };
    let mut group_values = vec![0.0; rows * dim];
    let mut group_scores = vec![f32::NEG_INFINITY; rows * dim];
    for group in 0..groups {
        group_values.fill(0.0);
        group_scores.fill(f32::NEG_INFINITY);
        for position in 0..ratio {
            if overlap && group > 0 {
                let token = (group - 1) * ratio + position;
                let source = token * projected_dim;
                let target = position * dim;
                group_values[target..target + dim].copy_from_slice(&kv[source..source + dim]);
                for column in 0..dim {
                    group_scores[target + column] =
                        scores[source + column] + ape[position * projected_dim + column];
                }
            }

            let token = group * ratio + position;
            let source_half = if overlap { dim } else { 0 };
            let source = token * projected_dim + source_half;
            let target_row = if overlap { ratio + position } else { position };
            let target = target_row * dim;
            group_values[target..target + dim].copy_from_slice(&kv[source..source + dim]);
            let ape_source = position * projected_dim + source_half;
            for column in 0..dim {
                group_scores[target + column] = scores[source + column] + ape[ape_source + column];
            }
        }
        softmax_weighted_sum(
            &group_values,
            &group_scores,
            rows,
            dim,
            &mut output[group * dim..(group + 1) * dim],
        );
    }
    Ok(output)
}

/// Learned indexer top-k selection for a complete prefill.
///
/// `query` is `[tokens, heads, dim]`, `compressed_kv` is `[groups, dim]`,
/// and `head_weights` is `[tokens, heads]`. Returned indices use the GGUF
/// attention-cache `offset`; causally unavailable slots are `-1`.
pub fn indexer_topk_prefill(
    query: &[f32],
    compressed_kv: &[f32],
    head_weights: &[f32],
    tokens: usize,
    heads: usize,
    dim: usize,
    ratio: usize,
    top_k: usize,
    offset: usize,
) -> Result<Vec<Vec<i32>>, CompressorError> {
    if ratio == 0 || dim == 0 {
        return Err(CompressorError::EmptyShape);
    }
    let expected_query = tokens.saturating_mul(heads).saturating_mul(dim);
    if query.len() != expected_query {
        return Err(CompressorError::QueryShape {
            expected: expected_query,
            actual: query.len(),
        });
    }
    let expected_weights = tokens.saturating_mul(heads);
    if head_weights.len() != expected_weights {
        return Err(CompressorError::WeightShape {
            expected: expected_weights,
            actual: head_weights.len(),
        });
    }
    if compressed_kv.len() % dim != 0 {
        return Err(CompressorError::KvShape {
            dim,
            actual: compressed_kv.len(),
        });
    }
    if query
        .iter()
        .chain(compressed_kv)
        .chain(head_weights)
        .any(|value| !value.is_finite())
    {
        return Err(CompressorError::NonFinite);
    }

    let groups = compressed_kv.len() / dim;
    let width = top_k.min(groups);
    let mut rows = Vec::with_capacity(tokens);
    for token in 0..tokens {
        let completed = ((token + 1) / ratio).min(groups);
        let mut scores: Vec<(usize, f32)> = (0..groups)
            .map(|group| {
                let mut score = 0.0;
                for head in 0..heads {
                    let query_offset = (token * heads + head) * dim;
                    let kv_offset = group * dim;
                    let dot = (0..dim)
                        .map(|column| {
                            query[query_offset + column] * compressed_kv[kv_offset + column]
                        })
                        .sum::<f32>()
                        .max(0.0);
                    score += dot * head_weights[token * heads + head];
                }
                (
                    group,
                    if group < completed {
                        score
                    } else {
                        f32::NEG_INFINITY
                    },
                )
            })
            .collect();
        scores.sort_unstable_by(|left, right| {
            right
                .1
                .total_cmp(&left.1)
                .then_with(|| left.0.cmp(&right.0))
        });
        rows.push(
            scores
                .into_iter()
                .take(width)
                .map(|(group, score)| {
                    if score.is_infinite() && score.is_sign_negative() {
                        -1
                    } else {
                        (offset + group) as i32
                    }
                })
                .collect(),
        );
    }
    Ok(rows)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn non_overlapping_prefill_uses_per_feature_softmax() {
        let kv = [1.0, 10.0, 3.0, 30.0, 5.0, 50.0, 7.0, 70.0, 99.0, 99.0];
        let scores = [0.0; 10];
        let ape = [0.0; 4];
        let output = compress_prefill_projected(&kv, &scores, &ape, 5, 2, 2, false).unwrap();
        assert_eq!(output, vec![2.0, 20.0, 6.0, 60.0]);
    }

    #[test]
    fn ratio_four_overlap_shifts_first_half_from_previous_group() {
        let ratio = 2;
        let dim = 1;
        // Per token: [overlap-half, normal-half].
        let kv = [10.0, 1.0, 20.0, 3.0, 30.0, 5.0, 40.0, 7.0];
        let scores = [0.0; 8];
        let ape = [0.0; 4];
        let output = compress_prefill_projected(&kv, &scores, &ape, 4, ratio, dim, true).unwrap();
        // First group has no preceding overlap. Second pools 10,20 with 5,7.
        assert_eq!(output, vec![2.0, 10.5]);
    }

    #[test]
    fn learned_indexer_is_causal_and_uses_unclamped_head_weights() {
        // Four tokens, two heads, scalar queries; two compressed groups.
        let query = [1.0, 1.0, 1.0, 1.0, 2.0, 1.0, 1.0, 3.0];
        let kv = [1.0, 2.0];
        let weights = [1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0];
        let rows = indexer_topk_prefill(&query, &kv, &weights, 4, 2, 1, 2, 2, 128).unwrap();
        assert_eq!(rows[0], vec![-1, -1]);
        assert_eq!(rows[1], vec![128, -1]);
        assert_eq!(rows[2], vec![128, -1]);
        assert_eq!(rows[3], vec![128, 129]);
    }

    #[test]
    fn malformed_projection_and_index_shapes_fail_closed() {
        assert_eq!(
            compress_prefill_projected(&[0.0; 3], &[0.0; 4], &[0.0; 2], 2, 2, 1, false)
                .unwrap_err(),
            CompressorError::ProjectionShape {
                expected: 2,
                actual: 3
            }
        );
        assert_eq!(
            indexer_topk_prefill(&[0.0; 3], &[0.0], &[0.0], 1, 1, 1, 4, 1, 0).unwrap_err(),
            CompressorError::QueryShape {
                expected: 1,
                actual: 3
            }
        );
    }
}
