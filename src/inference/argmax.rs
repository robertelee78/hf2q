//! Shared fail-closed handling for GPU argmax results.
//!
//! The Metal reduction reports both the winning index and value.  Every
//! generative consumer must validate the value before accepting the index:
//! non-finite model output is an inference error, never token zero.  Reading
//! the value adds no synchronization because these helpers are called only
//! after the same terminal wait that makes the index host-visible.

use anyhow::{Context, Result};
use mlx_native::MlxBuffer;

pub(crate) fn validate_finite_logits(values: &[f32], context: &str) -> Result<()> {
    if let Some((offset, value)) = values
        .iter()
        .copied()
        .enumerate()
        .find(|(_, value)| !value.is_finite())
    {
        anyhow::bail!("{context}: non-finite model logit at offset {offset}: {value}");
    }
    Ok(())
}

fn validate_argmax_row(index: u32, value: f32, vocab_size: u32, context: &str) -> Result<()> {
    anyhow::ensure!(vocab_size > 0, "{context}: vocabulary is empty");
    anyhow::ensure!(
        value.is_finite(),
        "{context}: GPU argmax produced a non-finite winning value: {value}"
    );
    anyhow::ensure!(
        index < vocab_size,
        "{context}: GPU argmax index {index} exceeds vocabulary size {vocab_size}"
    );
    Ok(())
}

pub(crate) fn read_finite_argmax_one(
    indices: &MlxBuffer,
    values: &MlxBuffer,
    vocab_size: u32,
    context: &str,
) -> Result<u32> {
    let indices = indices
        .as_slice::<u32>()
        .with_context(|| format!("{context}: read GPU argmax index"))?;
    let values = values
        .as_slice::<f32>()
        .with_context(|| format!("{context}: read GPU argmax value"))?;
    anyhow::ensure!(
        !indices.is_empty() && !values.is_empty(),
        "{context}: GPU argmax returned an empty result"
    );
    validate_argmax_row(indices[0], values[0], vocab_size, context)?;
    Ok(indices[0])
}

pub(crate) fn read_finite_argmax_batch(
    indices: &MlxBuffer,
    values: &MlxBuffer,
    rows: usize,
    vocab_size: u32,
    context: &str,
) -> Result<Vec<u32>> {
    let indices = indices
        .as_slice::<u32>()
        .with_context(|| format!("{context}: read GPU argmax indices"))?;
    let values = values
        .as_slice::<f32>()
        .with_context(|| format!("{context}: read GPU argmax values"))?;
    anyhow::ensure!(
        indices.len() >= rows && values.len() >= rows,
        "{context}: GPU argmax returned {} indices and {} values for {rows} rows",
        indices.len(),
        values.len()
    );
    for row in 0..rows {
        validate_argmax_row(indices[row], values[row], vocab_size, context)
            .with_context(|| format!("{context}: row {row}"))?;
    }
    Ok(indices[..rows].to_vec())
}

#[cfg(test)]
mod tests {
    use super::{validate_argmax_row, validate_finite_logits};

    #[test]
    fn argmax_row_rejects_nonfinite_values_and_invalid_indices() {
        assert!(validate_argmax_row(7, 1.25, 8, "test").is_ok());
        for value in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            assert!(validate_argmax_row(0, value, 8, "test").is_err());
        }
        assert!(validate_argmax_row(8, 1.25, 8, "test").is_err());
        assert!(validate_argmax_row(0, 1.25, 0, "test").is_err());
    }

    #[test]
    fn raw_logit_boundary_rejects_any_nonfinite_value() {
        assert!(validate_finite_logits(&[0.0, -1.0, 3.0], "test").is_ok());
        for value in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            assert!(validate_finite_logits(&[0.0, value, 3.0], "test").is_err());
        }
    }
}
