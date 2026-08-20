use std::collections::BTreeSet;
use std::fmt;

use safetensors::tensor::{Dtype, Metadata, TensorInfo};
use serde::de::{MapAccess, Visitor};
use serde::{Deserialize, Deserializer};

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct StrictTensorInfo {
    dtype: Dtype,
    shape: Vec<usize>,
    data_offsets: (usize, usize),
}

struct UniqueHeader(Vec<(String, TensorInfo)>);

struct UniqueHeaderVisitor;

struct UniqueMetadata;

struct UniqueMetadataVisitor;

impl<'de> Visitor<'de> for UniqueMetadataVisitor {
    type Value = UniqueMetadata;

    fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("a safetensors string metadata map with unique keys")
    }

    fn visit_map<M>(self, mut map: M) -> Result<Self::Value, M::Error>
    where
        M: MapAccess<'de>,
    {
        let mut seen = BTreeSet::new();
        while let Some((key, _value)) = map.next_entry::<String, String>()? {
            if !seen.insert(key.clone()) {
                return Err(serde::de::Error::custom(format!(
                    "duplicate safetensors metadata key {key}"
                )));
            }
        }
        Ok(UniqueMetadata)
    }
}

impl<'de> Deserialize<'de> for UniqueMetadata {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        deserializer.deserialize_map(UniqueMetadataVisitor)
    }
}

impl<'de> Visitor<'de> for UniqueHeaderVisitor {
    type Value = UniqueHeader;

    fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("a safetensors header with unique top-level keys")
    }

    fn visit_map<M>(self, mut map: M) -> Result<Self::Value, M::Error>
    where
        M: MapAccess<'de>,
    {
        let mut seen = BTreeSet::new();
        let mut tensors = Vec::new();
        while let Some(name) = map.next_key::<String>()? {
            if !seen.insert(name.clone()) {
                return Err(serde::de::Error::custom(format!(
                    "duplicate safetensors header key {name}"
                )));
            }
            if name == "__metadata__" {
                map.next_value::<UniqueMetadata>()?;
                continue;
            }
            let info = map.next_value::<StrictTensorInfo>()?;
            tensors.push((
                name,
                TensorInfo {
                    dtype: info.dtype,
                    shape: info.shape,
                    data_offsets: info.data_offsets,
                },
            ));
        }
        Ok(UniqueHeader(tensors))
    }
}

impl<'de> Deserialize<'de> for UniqueHeader {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        deserializer.deserialize_map(UniqueHeaderVisitor)
    }
}

pub(super) fn parse_unique_header(bytes: &[u8]) -> anyhow::Result<Metadata> {
    let mut deserializer = serde_json::Deserializer::from_slice(bytes);
    let mut header = UniqueHeader::deserialize(&mut deserializer)?;
    deserializer.end()?;
    header
        .0
        .sort_by(|(_, left), (_, right)| left.data_offsets.cmp(&right.data_offsets));
    Metadata::new(None, header.0).map_err(Into::into)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parser_rejects_duplicate_tensor_keys_and_unknown_fields() {
        let duplicate = br#"{
            "a": {"dtype": "BF16", "shape": [1], "data_offsets": [0, 2]},
            "a": {"dtype": "BF16", "shape": [1], "data_offsets": [0, 2]}
        }"#;
        assert!(parse_unique_header(duplicate)
            .unwrap_err()
            .to_string()
            .contains("duplicate safetensors header key a"));

        let unknown = br#"{
            "a": {"dtype": "BF16", "shape": [1], "data_offsets": [0, 2], "extra": 1}
        }"#;
        assert!(parse_unique_header(unknown)
            .unwrap_err()
            .to_string()
            .contains("unknown field"));

        let valid = br#"{
            "a": {"dtype": "BF16", "shape": [1], "data_offsets": [0, 2]}
        }"#;
        assert!(parse_unique_header(valid).is_ok());
    }
}
