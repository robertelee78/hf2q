//! Exact CPU references for DeepSeek-V4 sparse-attention index construction.
//!
//! Decode uses a circular sliding-window cache plus optional compressed-KV
//! positions. The learned attention sink participates in softmax normalization
//! but contributes no value vector.

use thiserror::Error;

#[derive(Debug, Error, Eq, PartialEq)]
pub enum AttentionError {
    #[error("window_size and compression ratio must be greater than zero")]
    EmptyWindow,
    #[error("incremental index construction requires seqlen=1, got {seqlen}")]
    IncrementalSequence { seqlen: usize },
    #[error("query length {actual} does not match heads*dim ({expected})")]
    QueryShape { expected: usize, actual: usize },
    #[error("KV length {actual} is not divisible by dim {dim}")]
    KvShape { dim: usize, actual: usize },
    #[error("expected {expected} attention sinks, got {actual}")]
    SinkShape { expected: usize, actual: usize },
    #[error("sparse-attention index {index} is outside -1 or 0..{kv_count}")]
    InvalidIndex { index: i32, kv_count: usize },
    #[error("sparse-attention input contains a non-finite value")]
    NonFinite,
}

#[derive(Debug, Eq, PartialEq)]
pub(super) struct CompressedAttentionIndexPlan {
    pub(super) storage: Vec<i32>,
    pub(super) attention_width: usize,
    pub(super) indexer_output_offset: Option<usize>,
}

/// Build a compact sparse-attention prefix while retaining fixed top-k
/// storage for the ratio-four indexer output.
///
/// Masked `-1` slots do not participate in the official softmax, so dropping
/// only the trailing padding preserves the valid index order and arithmetic.
pub(super) fn compressed_attention_index_plan(
    ratio: usize,
    window_size: usize,
    index_top_k: usize,
    position: usize,
) -> Result<CompressedAttentionIndexPlan, AttentionError> {
    let mut storage = window_indices(window_size, 1, position)?
        .into_iter()
        .next()
        .expect("one-token window plan must contain one row");
    storage.retain(|&index| index >= 0);
    let window_valid = storage.len();
    if ratio == 4 {
        let selected = ((position + 1) / ratio).min(index_top_k);
        let attention_width = window_valid + selected;
        storage.resize(window_valid + index_top_k, -1);
        return Ok(CompressedAttentionIndexPlan {
            storage,
            attention_width,
            indexer_output_offset: Some(window_valid),
        });
    }
    storage.extend(
        compressed_indices(ratio, 1, position, window_size)?
            .into_iter()
            .next()
            .expect("one-token compressed plan must contain one row"),
    );
    Ok(CompressedAttentionIndexPlan {
        attention_width: storage.len(),
        storage,
        indexer_output_offset: None,
    })
}

/// Sliding-window positions for every query in a prefill, or the one query in
/// an incremental decode call. `-1` is the official masked-position sentinel.
pub fn window_indices(
    window_size: usize,
    seqlen: usize,
    start_pos: usize,
) -> Result<Vec<Vec<i32>>, AttentionError> {
    if window_size == 0 {
        return Err(AttentionError::EmptyWindow);
    }
    if start_pos > 0 && seqlen != 1 {
        return Err(AttentionError::IncrementalSequence { seqlen });
    }
    if start_pos >= window_size - 1 {
        let cursor = start_pos % window_size;
        let row = ((cursor + 1)..window_size)
            .chain(0..=cursor)
            .map(|value| value as i32)
            .collect();
        return Ok(vec![row]);
    }
    if start_pos > 0 {
        let mut row: Vec<i32> = (0..=start_pos).map(|value| value as i32).collect();
        row.resize(window_size, -1);
        return Ok(vec![row]);
    }

    let width = seqlen.min(window_size);
    Ok((0..seqlen)
        .map(|query| {
            let first = query.saturating_sub(window_size - 1);
            (0..width)
                .map(|column| {
                    let position = first + column;
                    if position > query {
                        -1
                    } else {
                        position as i32
                    }
                })
                .collect()
        })
        .collect())
}

/// Completed compressed-KV positions. Prefill masks the current incomplete
/// compression group; decode exposes all groups completed through `start_pos`.
pub fn compressed_indices(
    ratio: usize,
    seqlen: usize,
    start_pos: usize,
    offset: usize,
) -> Result<Vec<Vec<i32>>, AttentionError> {
    if ratio == 0 {
        return Err(AttentionError::EmptyWindow);
    }
    let groups = (start_pos + seqlen) / ratio;
    Ok((0..seqlen)
        .map(|query| {
            let completed = (start_pos + query + 1) / ratio;
            (0..groups)
                .map(|group| {
                    if group >= completed {
                        -1
                    } else {
                        (offset + group) as i32
                    }
                })
                .collect()
        })
        .collect())
}

/// Raw-window indices for a multi-row append prefill whose compact KV input
/// is `[prior physical ring slots][current chunk rows]`.
///
/// Prior rows retain their physical circular-cache indices. Current rows use
/// the appended portion, so overwriting the live ring later in the command
/// buffer cannot change any query's causal view.
pub(super) fn append_prefill_window_indices(
    window_size: usize,
    seqlen: usize,
    start_pos: usize,
) -> Result<Vec<Vec<i32>>, AttentionError> {
    if window_size == 0 {
        return Err(AttentionError::EmptyWindow);
    }
    let width = (start_pos + seqlen).min(window_size);
    Ok((0..seqlen)
        .map(|query| {
            let absolute = start_pos + query;
            let first = absolute.saturating_sub(window_size - 1);
            let mut row = (first..=absolute)
                .map(|position| {
                    if position < start_pos {
                        (position % window_size) as i32
                    } else {
                        (window_size + position - start_pos) as i32
                    }
                })
                .collect::<Vec<_>>();
            row.resize(width, -1);
            row
        })
        .collect())
}

/// Sparse attention for one query token. `query` is `[heads, dim]`, `kv` is
/// `[positions, dim]`, and one index row is shared by every head.
pub fn sparse_attention_reference(
    query: &[f32],
    kv: &[f32],
    heads: usize,
    dim: usize,
    sinks: &[f32],
    indices: &[i32],
    scale: f32,
) -> Result<Vec<f32>, AttentionError> {
    let expected_query = heads.saturating_mul(dim);
    if query.len() != expected_query {
        return Err(AttentionError::QueryShape {
            expected: expected_query,
            actual: query.len(),
        });
    }
    if dim == 0 || kv.len() % dim != 0 {
        return Err(AttentionError::KvShape {
            dim,
            actual: kv.len(),
        });
    }
    if sinks.len() != heads {
        return Err(AttentionError::SinkShape {
            expected: heads,
            actual: sinks.len(),
        });
    }
    if !scale.is_finite()
        || query.iter().any(|value| !value.is_finite())
        || kv.iter().any(|value| !value.is_finite())
        || sinks.iter().any(|value| !value.is_finite())
    {
        return Err(AttentionError::NonFinite);
    }
    let kv_count = kv.len() / dim;
    for &index in indices {
        if index < -1 || index >= kv_count as i32 {
            return Err(AttentionError::InvalidIndex { index, kv_count });
        }
    }

    let mut output = vec![0.0; expected_query];
    let mut scores = Vec::with_capacity(indices.len());
    for head in 0..heads {
        scores.clear();
        let query_row = &query[head * dim..(head + 1) * dim];
        let mut max_score = sinks[head];
        for &index in indices {
            if index == -1 {
                scores.push(None);
                continue;
            }
            let kv_row = &kv[index as usize * dim..(index as usize + 1) * dim];
            let score = query_row
                .iter()
                .zip(kv_row)
                .map(|(left, right)| left * right)
                .sum::<f32>()
                * scale;
            max_score = max_score.max(score);
            scores.push(Some(score));
        }
        let mut denominator = (sinks[head] - max_score).exp();
        for (slot, score) in scores.iter().enumerate() {
            let Some(score) = score else { continue };
            let weight = (*score - max_score).exp();
            denominator += weight;
            let index = indices[slot] as usize;
            for column in 0..dim {
                output[head * dim + column] += weight * kv[index * dim + column];
            }
        }
        for value in &mut output[head * dim..(head + 1) * dim] {
            *value /= denominator;
        }
    }
    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prefill_window_is_causal_and_then_slides() {
        assert_eq!(
            window_indices(4, 6, 0).unwrap(),
            vec![
                vec![0, -1, -1, -1],
                vec![0, 1, -1, -1],
                vec![0, 1, 2, -1],
                vec![0, 1, 2, 3],
                vec![1, 2, 3, 4],
                vec![2, 3, 4, 5],
            ]
        );
    }

    #[test]
    fn decode_window_tracks_circular_cache_order() {
        assert_eq!(window_indices(4, 1, 2).unwrap(), vec![vec![0, 1, 2, -1]]);
        assert_eq!(window_indices(4, 1, 5).unwrap(), vec![vec![2, 3, 0, 1]]);
    }

    #[test]
    fn compressed_indices_exclude_incomplete_groups() {
        let got = compressed_indices(4, 10, 0, 10).unwrap();
        assert_eq!(got[2], vec![-1, -1]);
        assert_eq!(got[3], vec![10, -1]);
        assert_eq!(got[7], vec![10, 11]);
        assert_eq!(got[9], vec![10, 11]);
        assert_eq!(compressed_indices(4, 1, 7, 10).unwrap(), vec![vec![10, 11]]);
        assert_eq!(
            compressed_indices(4, 3, 6, 10).unwrap(),
            vec![vec![10, -1], vec![10, 11], vec![10, 11]]
        );
    }

    #[test]
    fn append_prefill_window_keeps_prior_ring_and_current_rows_distinct() {
        assert_eq!(
            append_prefill_window_indices(4, 3, 3).unwrap(),
            vec![vec![0, 1, 2, 4], vec![1, 2, 4, 5], vec![2, 4, 5, 6]]
        );
        assert_eq!(
            append_prefill_window_indices(4, 2, 6).unwrap(),
            vec![vec![3, 0, 1, 4], vec![0, 1, 4, 5]]
        );
    }

    #[test]
    fn compressed_attention_plan_drops_only_masked_padding() {
        let first = compressed_attention_index_plan(4, 128, 512, 0).unwrap();
        assert_eq!(first.attention_width, 1);
        assert_eq!(first.indexer_output_offset, Some(1));
        assert_eq!(first.storage.len(), 513);
        assert_eq!(&first.storage[..3], &[0, -1, -1]);

        let ratio4 = compressed_attention_index_plan(4, 128, 512, 130).unwrap();
        assert_eq!(ratio4.attention_width, 160);
        assert_eq!(ratio4.indexer_output_offset, Some(128));
        assert_eq!(ratio4.storage.len(), 640);

        let ratio128 = compressed_attention_index_plan(128, 128, 512, 130).unwrap();
        assert_eq!(ratio128.attention_width, 129);
        assert_eq!(ratio128.indexer_output_offset, None);
        assert_eq!(ratio128.storage.last(), Some(&128));
    }

    #[test]
    fn attention_sink_adds_probability_mass_without_a_value() {
        let output =
            sparse_attention_reference(&[1.0], &[1.0, 2.0], 1, 1, &[0.0], &[0, 1], 1.0).unwrap();
        let expected =
            (1.0_f32.exp() + 2.0 * 2.0_f32.exp()) / (0.0_f32.exp() + 1.0_f32.exp() + 2.0_f32.exp());
        assert!((output[0] - expected).abs() < 1e-6);

        let sink_dominated =
            sparse_attention_reference(&[1.0], &[1.0], 1, 1, &[20.0], &[0], 1.0).unwrap();
        assert!(sink_dominated[0] < 1e-7);
    }

    #[test]
    fn invalid_incremental_shapes_and_indices_fail_closed() {
        assert_eq!(
            window_indices(128, 2, 1).unwrap_err(),
            AttentionError::IncrementalSequence { seqlen: 2 }
        );
        assert_eq!(
            sparse_attention_reference(&[1.0], &[1.0], 1, 1, &[0.0], &[1], 1.0).unwrap_err(),
            AttentionError::InvalidIndex {
                index: 1,
                kv_count: 1
            }
        );
    }
}
