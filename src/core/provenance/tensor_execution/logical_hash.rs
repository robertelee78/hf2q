use sha2::{Digest, Sha256};
use thiserror::Error;

pub const LOGICAL_F32_HASH_ENCODING: &str = "hf2q-framed-f32-le-v1";

#[derive(Debug, Error, PartialEq, Eq)]
pub enum LogicalF32HashError {
    #[error("logical F32 shape is empty or contains a zero dimension")]
    InvalidShape,
    #[error("logical F32 element count overflow")]
    ElementCountOverflow,
    #[error("logical F32 stream has {observed} values, expected {expected}")]
    ElementCountMismatch { expected: u64, observed: u64 },
}

/// Incremental, shape-bound logical F32 hasher.
///
/// The digest frames the encoding name, rank, dimensions, element count, and
/// every exact `f32::to_bits()` value in little-endian order. It therefore
/// cannot alias differently shaped tensors or ambiguous byte concatenations.
#[derive(Debug, Clone)]
pub struct LogicalF32Hasher {
    hasher: Sha256,
    expected: u64,
    observed: u64,
}

impl LogicalF32Hasher {
    pub fn new(shape_outermost_first: &[u64]) -> Result<Self, LogicalF32HashError> {
        if shape_outermost_first.is_empty()
            || shape_outermost_first
                .iter()
                .any(|dimension| *dimension == 0)
        {
            return Err(LogicalF32HashError::InvalidShape);
        }
        let expected = shape_outermost_first
            .iter()
            .try_fold(1_u64, |count, dimension| count.checked_mul(*dimension))
            .ok_or(LogicalF32HashError::ElementCountOverflow)?;
        let mut hasher = Sha256::new();
        hasher.update(LOGICAL_F32_HASH_ENCODING.as_bytes());
        hasher.update([0]);
        hasher.update(
            u64::try_from(shape_outermost_first.len())
                .map_err(|_| LogicalF32HashError::ElementCountOverflow)?
                .to_le_bytes(),
        );
        for dimension in shape_outermost_first {
            hasher.update(dimension.to_le_bytes());
        }
        hasher.update(expected.to_le_bytes());
        Ok(Self {
            hasher,
            expected,
            observed: 0,
        })
    }

    pub fn update(&mut self, values: &[f32]) -> Result<(), LogicalF32HashError> {
        let added =
            u64::try_from(values.len()).map_err(|_| LogicalF32HashError::ElementCountOverflow)?;
        let observed = self
            .observed
            .checked_add(added)
            .ok_or(LogicalF32HashError::ElementCountOverflow)?;
        if observed > self.expected {
            return Err(LogicalF32HashError::ElementCountMismatch {
                expected: self.expected,
                observed,
            });
        }
        for value in values {
            self.hasher.update(value.to_bits().to_le_bytes());
        }
        self.observed = observed;
        Ok(())
    }

    pub fn finalize(self) -> Result<String, LogicalF32HashError> {
        if self.observed != self.expected {
            return Err(LogicalF32HashError::ElementCountMismatch {
                expected: self.expected,
                observed: self.observed,
            });
        }
        Ok(hex::encode(self.hasher.finalize()))
    }
}

pub fn logical_f32_sha256(
    shape_outermost_first: &[u64],
    values: &[f32],
) -> Result<String, LogicalF32HashError> {
    let mut hasher = LogicalF32Hasher::new(shape_outermost_first)?;
    hasher.update(values)?;
    hasher.finalize()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chunking_is_invariant_but_shape_and_bits_are_bound() {
        let values = [1.0_f32, -0.0, f32::from_bits(0x7fc0_0001), 4.0];
        let whole = logical_f32_sha256(&[2, 2], &values).unwrap();
        let mut chunked = LogicalF32Hasher::new(&[2, 2]).unwrap();
        chunked.update(&values[..1]).unwrap();
        chunked.update(&values[1..]).unwrap();
        assert_eq!(chunked.finalize().unwrap(), whole);
        assert_ne!(logical_f32_sha256(&[1, 4], &values).unwrap(), whole);

        let mut plus_zero = values;
        plus_zero[1] = 0.0;
        assert_ne!(logical_f32_sha256(&[2, 2], &plus_zero).unwrap(), whole);
    }

    #[test]
    fn invalid_shape_and_stream_lengths_fail_closed() {
        assert_eq!(
            LogicalF32Hasher::new(&[]).unwrap_err(),
            LogicalF32HashError::InvalidShape
        );
        let mut short = LogicalF32Hasher::new(&[2]).unwrap();
        short.update(&[1.0]).unwrap();
        assert!(matches!(
            short.finalize(),
            Err(LogicalF32HashError::ElementCountMismatch { .. })
        ));
        let mut long = LogicalF32Hasher::new(&[1]).unwrap();
        assert!(matches!(
            long.update(&[1.0, 2.0]),
            Err(LogicalF32HashError::ElementCountMismatch { .. })
        ));
    }
}
