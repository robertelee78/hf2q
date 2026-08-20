use std::collections::BTreeMap;
use std::fmt;

use anyhow::{ensure, Result};
use serde::de::{MapAccess, SeqAccess, Visitor};
use serde::{Deserialize, Deserializer};

use crate::convert::arch::qwen35_dense::is_qwen35_dense_vision_source_tensor;
use crate::intelligence::dynamic_allocator::producer::{
    NonVariableDisposition, TensorPartitionManifest,
};

use super::types::SourcePrecisionDisposition;

struct UniqueJsonValue(serde_json::Value);

struct UniqueJsonValueVisitor;

impl<'de> Visitor<'de> for UniqueJsonValueVisitor {
    type Value = UniqueJsonValue;

    fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("JSON with unique object keys")
    }

    fn visit_bool<E>(self, value: bool) -> Result<Self::Value, E> {
        Ok(UniqueJsonValue(value.into()))
    }

    fn visit_i64<E>(self, value: i64) -> Result<Self::Value, E> {
        Ok(UniqueJsonValue(value.into()))
    }

    fn visit_u64<E>(self, value: u64) -> Result<Self::Value, E> {
        Ok(UniqueJsonValue(value.into()))
    }

    fn visit_f64<E>(self, value: f64) -> Result<Self::Value, E>
    where
        E: serde::de::Error,
    {
        let number = serde_json::Number::from_f64(value)
            .ok_or_else(|| E::custom("config contains a non-finite number"))?;
        Ok(UniqueJsonValue(number.into()))
    }

    fn visit_str<E>(self, value: &str) -> Result<Self::Value, E> {
        Ok(UniqueJsonValue(value.into()))
    }

    fn visit_string<E>(self, value: String) -> Result<Self::Value, E> {
        Ok(UniqueJsonValue(value.into()))
    }

    fn visit_none<E>(self) -> Result<Self::Value, E> {
        Ok(UniqueJsonValue(serde_json::Value::Null))
    }

    fn visit_unit<E>(self) -> Result<Self::Value, E> {
        Ok(UniqueJsonValue(serde_json::Value::Null))
    }

    fn visit_some<D>(self, deserializer: D) -> Result<Self::Value, D::Error>
    where
        D: Deserializer<'de>,
    {
        UniqueJsonValue::deserialize(deserializer)
    }

    fn visit_seq<A>(self, mut sequence: A) -> Result<Self::Value, A::Error>
    where
        A: SeqAccess<'de>,
    {
        let mut values = Vec::new();
        while let Some(value) = sequence.next_element::<UniqueJsonValue>()? {
            values.push(value.0);
        }
        Ok(UniqueJsonValue(values.into()))
    }

    fn visit_map<A>(self, mut map: A) -> Result<Self::Value, A::Error>
    where
        A: MapAccess<'de>,
    {
        let mut values = serde_json::Map::new();
        while let Some((key, value)) = map.next_entry::<String, UniqueJsonValue>()? {
            if values.insert(key.clone(), value.0).is_some() {
                return Err(serde::de::Error::custom(format!(
                    "duplicate config key {key}"
                )));
            }
        }
        Ok(UniqueJsonValue(values.into()))
    }
}

impl<'de> Deserialize<'de> for UniqueJsonValue {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        deserializer.deserialize_any(UniqueJsonValueVisitor)
    }
}

pub(super) fn parse_unique_qwen_config(bytes: &[u8]) -> Result<serde_json::Value> {
    let mut deserializer = serde_json::Deserializer::from_slice(bytes);
    let value = UniqueJsonValue::deserialize(&mut deserializer)?;
    deserializer.end()?;
    Ok(value.0)
}

pub(super) fn source_dispositions(
    partition: &TensorPartitionManifest,
) -> Result<BTreeMap<String, SourcePrecisionDisposition>> {
    let mut dispositions = BTreeMap::new();
    for unit in &partition.variable_units {
        for member in &unit.members {
            ensure!(
                dispositions
                    .insert(member.name.clone(), SourcePrecisionDisposition::Variable)
                    .is_none(),
                "duplicate variable source tensor {}",
                member.name
            );
        }
    }
    for tensor in &partition.non_variable_tensors {
        let disposition = match tensor.disposition {
            NonVariableDisposition::Fixed => SourcePrecisionDisposition::Fixed,
            NonVariableDisposition::Protected => SourcePrecisionDisposition::Protected,
            NonVariableDisposition::Excluded => SourcePrecisionDisposition::Excluded,
        };
        ensure!(
            dispositions
                .insert(tensor.source.name.clone(), disposition)
                .is_none(),
            "duplicate non-variable source tensor {}",
            tensor.source.name
        );
    }
    Ok(dispositions)
}

pub(super) fn validate_teacher_dispositions(
    dispositions: &BTreeMap<String, SourcePrecisionDisposition>,
) -> Result<()> {
    for (name, disposition) in dispositions {
        if is_qwen35_dense_vision_source_tensor(name) {
            ensure!(
                *disposition == SourcePrecisionDisposition::Excluded,
                "vision source tensor {name} must be explicitly excluded"
            );
        } else {
            ensure!(
                *disposition != SourcePrecisionDisposition::Excluded,
                "text source tensor {name} cannot be excluded from the teacher snapshot"
            );
        }
        if name.starts_with("mtp.") {
            ensure!(
                matches!(
                    disposition,
                    SourcePrecisionDisposition::Fixed | SourcePrecisionDisposition::Protected
                ),
                "MTP source tensor {name} must be fixed or protected"
            );
        }
    }
    ensure!(
        dispositions
            .get("lm_head.weight")
            .is_some_and(|value| *value != SourcePrecisionDisposition::Excluded),
        "dense-Qwen source teacher requires an untied lm_head.weight"
    );
    Ok(())
}

pub(super) fn validate_dense_qwen_source_config(config: &serde_json::Value) -> Result<()> {
    let text = config.get("text_config").unwrap_or(config);
    let architecture = config
        .get("architectures")
        .and_then(serde_json::Value::as_array)
        .filter(|values| values.len() == 1)
        .and_then(|values| values[0].as_str());
    let exact_family_shape = match architecture {
        Some("Qwen3_5ForConditionalGeneration") => {
            config.get("model_type").and_then(serde_json::Value::as_str) == Some("qwen3_5")
                && config.get("text_config").is_some()
                && text.get("model_type").and_then(serde_json::Value::as_str)
                    == Some("qwen3_5_text")
        }
        Some("Qwen3_5ForCausalLM") => {
            config.get("text_config").is_none()
                && config.get("model_type").and_then(serde_json::Value::as_str)
                    == Some("qwen3_5_text")
        }
        _ => false,
    };
    ensure!(
        exact_family_shape,
        "source config is not the admitted dense-Qwen3.5 text family"
    );
    ensure!(
        config.get("quantization_config").is_none()
            && text.get("quantization_config").is_none()
            && text.get("num_experts").is_none()
            && text.get("moe_intermediate_size").is_none(),
        "source config contains quantized or MoE state"
    );
    Ok(())
}
