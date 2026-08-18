//! Bounded hostile-JSON preflight for signed metadata.

use std::collections::HashSet;
use std::fmt;

use serde::de::{DeserializeSeed, Error as _, MapAccess, SeqAccess, Visitor};

use super::TufVerifierError;

pub(super) const MAX_JSON_DEPTH: usize = 64;

pub(super) fn validate(bytes: &[u8], maximum: usize) -> Result<(), TufVerifierError> {
    if bytes.is_empty() || bytes.len() > maximum {
        return Err(TufVerifierError::MetadataSize);
    }
    let mut deserializer = serde_json::Deserializer::from_slice(bytes);
    Seed { depth: 0 }
        .deserialize(&mut deserializer)
        .map_err(|error| {
            if error.to_string().contains("duplicate signed-metadata key") {
                TufVerifierError::DuplicateJsonKey
            } else {
                TufVerifierError::MalformedMetadata
            }
        })?;
    deserializer
        .end()
        .map_err(|_| TufVerifierError::MalformedMetadata)
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
            return Err(D::Error::custom("signed-metadata nesting is too deep"));
        }
        deserializer.deserialize_any(self)
    }
}

impl<'de> Visitor<'de> for Seed {
    type Value = ();

    fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("bounded signed-metadata JSON")
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
    fn visit_seq<A>(self, mut sequence: A) -> Result<(), A::Error>
    where
        A: SeqAccess<'de>,
    {
        while sequence
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
                return Err(A::Error::custom("duplicate signed-metadata key"));
            }
            map.next_value_seed(Seed {
                depth: self.depth + 1,
            })?;
        }
        Ok(())
    }
}
