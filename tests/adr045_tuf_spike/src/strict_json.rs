//! Bounded duplicate-key and trailing-data rejection before candidate parsing.

use std::collections::HashSet;
use std::fmt;

use serde::de::{DeserializeSeed, Error as _, MapAccess, SeqAccess, Visitor};

use crate::model::SpikeError;

const MAX_JSON_DEPTH: usize = 64;

pub(crate) fn validate(bytes: &[u8], max_bytes: usize) -> Result<(), SpikeError> {
    if bytes.len() > max_bytes {
        return Err(SpikeError::MetadataTooLarge);
    }
    let mut de = serde_json::Deserializer::from_slice(bytes);
    Seed { depth: 0 }
        .deserialize(&mut de)
        .map_err(map_json_error)?;
    de.end().map_err(|_| SpikeError::MalformedMetadata)
}

fn map_json_error(error: serde_json::Error) -> SpikeError {
    if error.to_string().contains("duplicate JSON key") {
        SpikeError::DuplicateJsonKey
    } else {
        SpikeError::MalformedMetadata
    }
}

#[derive(Clone, Copy)]
struct Seed {
    depth: usize,
}

impl<'de> DeserializeSeed<'de> for Seed {
    type Value = ();

    fn deserialize<D>(self, deserializer: D) -> Result<Self::Value, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        if self.depth > MAX_JSON_DEPTH {
            return Err(D::Error::custom("JSON nesting exceeds spike limit"));
        }
        deserializer.deserialize_any(self)
    }
}

impl<'de> Visitor<'de> for Seed {
    type Value = ();

    fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("bounded JSON")
    }

    fn visit_bool<E>(self, _value: bool) -> Result<(), E> {
        Ok(())
    }

    fn visit_i64<E>(self, _value: i64) -> Result<(), E> {
        Ok(())
    }

    fn visit_u64<E>(self, _value: u64) -> Result<(), E> {
        Ok(())
    }

    fn visit_f64<E>(self, _value: f64) -> Result<(), E> {
        Ok(())
    }

    fn visit_str<E>(self, _value: &str) -> Result<(), E> {
        Ok(())
    }

    fn visit_string<E>(self, _value: String) -> Result<(), E> {
        Ok(())
    }

    fn visit_none<E>(self) -> Result<(), E> {
        Ok(())
    }

    fn visit_unit<E>(self) -> Result<(), E> {
        Ok(())
    }

    fn visit_some<D>(self, deserializer: D) -> Result<(), D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        Seed {
            depth: self.depth + 1,
        }
        .deserialize(deserializer)
    }

    fn visit_seq<A>(self, mut seq: A) -> Result<(), A::Error>
    where
        A: SeqAccess<'de>,
    {
        while seq
            .next_element_seed(Seed {
                depth: self.depth + 1,
            })?
            .is_some()
        {}
        Ok(())
    }

    fn visit_map<A>(self, mut map: A) -> Result<(), A::Error>
    where
        A: MapAccess<'de>,
    {
        let mut keys = HashSet::new();
        while let Some(key) = map.next_key::<String>()? {
            if !keys.insert(key) {
                return Err(A::Error::custom("duplicate JSON key"));
            }
            map.next_value_seed(Seed {
                depth: self.depth + 1,
            })?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::validate;
    use crate::model::SpikeError;

    #[test]
    fn rejects_duplicate_keys_and_trailing_json() {
        assert!(matches!(
            validate(br#"{"a":1,"a":2}"#, 64),
            Err(SpikeError::DuplicateJsonKey)
        ));
        assert!(matches!(
            validate(br#"{} {}"#, 64),
            Err(SpikeError::MalformedMetadata)
        ));
    }
}
