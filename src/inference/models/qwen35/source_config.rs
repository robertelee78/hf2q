//! Strict authenticated dense-Qwen source configuration projection.

use anyhow::{bail, ensure, Context, Result};
use serde_json::Value;

use super::{default_layer_types, Qwen35Config, Qwen35Variant};

const MAX_QWEN35_EVIDENCE_LAYERS: u32 = 256;

fn required_u32(config: &Value, key: &str) -> Result<u32> {
    u32::try_from(
        config
            .get(key)
            .and_then(Value::as_u64)
            .with_context(|| format!("authenticated Qwen config is missing {key}"))?,
    )
    .with_context(|| format!("authenticated Qwen config {key} exceeds u32"))
}

fn optional_u32(config: &Value, key: &str) -> Result<Option<u32>> {
    let Some(value) = config.get(key) else {
        return Ok(None);
    };
    let raw = value
        .as_u64()
        .with_context(|| format!("authenticated Qwen config {key} must be an unsigned integer"))?;
    Ok(Some(u32::try_from(raw).with_context(|| {
        format!("authenticated Qwen config {key} exceeds u32")
    })?))
}

fn optional_f64(config: &Value, key: &str) -> Result<Option<f64>> {
    let Some(value) = config.get(key) else {
        return Ok(None);
    };
    Ok(Some(value.as_f64().with_context(|| {
        format!("authenticated Qwen config {key} must be numeric")
    })?))
}

fn optional_bool(config: &Value, key: &str) -> Result<Option<bool>> {
    let Some(value) = config.get(key) else {
        return Ok(None);
    };
    Ok(Some(value.as_bool().with_context(|| {
        format!("authenticated Qwen config {key} must be boolean")
    })?))
}

pub(super) fn qwen35_config_from_authenticated_source(config: &Value) -> Result<Qwen35Config> {
    let text = config.get("text_config").unwrap_or(config);
    let rope_parameters = match text.get("rope_parameters") {
        Some(value) => Some(
            value
                .as_object()
                .context("authenticated Qwen config rope_parameters must be an object")?,
        ),
        None => None,
    };
    let hidden_size = required_u32(text, "hidden_size")?;
    let num_hidden_layers = required_u32(text, "num_hidden_layers")?;
    let num_attention_heads = required_u32(text, "num_attention_heads")?;
    let num_key_value_heads =
        optional_u32(text, "num_key_value_heads")?.unwrap_or(num_attention_heads);
    ensure!(
        num_attention_heads > 0 && hidden_size > 0,
        "authenticated Qwen dimensions must be positive"
    );
    let head_dim = optional_u32(text, "head_dim")?.unwrap_or(hidden_size / num_attention_heads);
    let linear_num_key_heads = required_u32(text, "linear_num_key_heads")?;
    let linear_num_value_heads = required_u32(text, "linear_num_value_heads")?;
    let linear_key_head_dim = required_u32(text, "linear_key_head_dim")?;
    let linear_value_head_dim = required_u32(text, "linear_value_head_dim")?;
    let linear_conv_kernel_dim = required_u32(text, "linear_conv_kernel_dim")?;
    let intermediate_size = required_u32(text, "intermediate_size")?;
    let max_position_embeddings = required_u32(text, "max_position_embeddings")?;
    let vocabulary_size = required_u32(text, "vocab_size")?;
    ensure!(
        num_hidden_layers > 0 && num_hidden_layers <= MAX_QWEN35_EVIDENCE_LAYERS,
        "authenticated Qwen layer count must be in 1..={MAX_QWEN35_EVIDENCE_LAYERS}"
    );
    ensure!(
        num_key_value_heads > 0
            && head_dim > 0
            && linear_num_key_heads > 0
            && linear_num_value_heads > 0
            && linear_num_value_heads % linear_num_key_heads == 0
            && linear_key_head_dim > 0
            && linear_value_head_dim > 0
            && linear_conv_kernel_dim > 0
            && intermediate_size > 0
            && max_position_embeddings > 0
            && vocabulary_size > 0,
        "authenticated Qwen dimensions must be positive and grouped-head geometry divisible"
    );
    ensure!(
        linear_key_head_dim == linear_value_head_dim,
        "dense Qwen evidence requires equal linear key/value head dimensions"
    );
    let full_attention_interval = optional_u32(text, "full_attention_interval")?.unwrap_or(4);
    ensure!(
        full_attention_interval > 0,
        "authenticated Qwen full_attention_interval must be positive"
    );
    let partial_rotary_factor = match optional_f64(text, "partial_rotary_factor")? {
        Some(value) => value,
        None => match rope_parameters.and_then(|rope| rope.get("partial_rotary_factor")) {
            Some(value) => value.as_f64().context(
                "authenticated Qwen config rope_parameters.partial_rotary_factor must be numeric",
            )?,
            None => 0.25,
        },
    } as f32;
    let rope_theta = match rope_parameters.and_then(|rope| rope.get("rope_theta")) {
        Some(value) => value
            .as_f64()
            .context("authenticated Qwen config rope_parameters.rope_theta must be numeric")?,
        None => optional_f64(text, "rope_theta")?.unwrap_or(10_000.0),
    } as f32 as f64;
    let mrope_interleaved = match rope_parameters.and_then(|rope| rope.get("mrope_interleaved")) {
        Some(value) => value.as_bool().context(
            "authenticated Qwen config rope_parameters.mrope_interleaved must be boolean",
        )?,
        None => optional_bool(text, "mrope_interleaved")?.unwrap_or(true),
    };
    let rotary_dim = (f64::from(head_dim) * f64::from(partial_rotary_factor)) as u32;
    let mrope_section = rope_parameters
        .and_then(|rope| rope.get("mrope_section"))
        .or_else(|| text.get("mrope_section"));
    let mut mrope = match mrope_section {
        Some(value) => Some(
            value
                .as_array()
                .context("authenticated Qwen config mrope_section must be an array")?
                .iter()
                .map(|value| {
                    u32::try_from(value.as_u64().context("negative mrope section")?)
                        .context("mrope section exceeds u32")
                })
                .collect::<Result<Vec<_>>>()?,
        ),
        None => None,
    }
    .unwrap_or_else(|| vec![11, 11, 10]);
    while mrope.len() < 4 {
        mrope.push(0);
    }
    if mrope.len() != 4 {
        bail!("authenticated Qwen mrope section must contain at most four values");
    }
    let mtp_num_hidden_layers = optional_u32(text, "mtp_num_hidden_layers")?.unwrap_or(0);
    let requested_mtp_dedicated = optional_bool(text, "mtp_use_dedicated_embeddings")?;
    let mtp_use_dedicated_embeddings = if mtp_num_hidden_layers == 0 {
        true
    } else {
        requested_mtp_dedicated.unwrap_or(false)
    };
    let rms_norm_eps = optional_f64(text, "rms_norm_eps")?.unwrap_or(1e-6) as f32;

    Ok(Qwen35Config {
        variant: Qwen35Variant::Dense,
        hidden_size,
        num_hidden_layers,
        num_attention_heads,
        num_key_value_heads,
        head_dim,
        linear_num_key_heads,
        linear_num_value_heads,
        linear_key_head_dim,
        linear_value_head_dim,
        linear_conv_kernel_dim,
        full_attention_interval,
        layer_types: default_layer_types(num_hidden_layers, full_attention_interval),
        partial_rotary_factor,
        rope_theta,
        rotary_dim,
        mrope_section: [mrope[0], mrope[1], mrope[2], mrope[3]],
        mrope_interleaved,
        rms_norm_eps,
        max_position_embeddings,
        vocab_size: vocabulary_size,
        attn_output_gate: match optional_bool(text, "attn_output_gate")? {
            Some(value) => value,
            None => optional_bool(text, "attention_output_gate")?.unwrap_or(true),
        },
        mtp_num_hidden_layers,
        mtp_use_dedicated_embeddings,
        intermediate_size: Some(intermediate_size),
        moe: None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn valid_config() -> Value {
        json!({
            "hidden_size": 256,
            "num_hidden_layers": 4,
            "num_attention_heads": 4,
            "num_key_value_heads": 1,
            "head_dim": 64,
            "linear_num_key_heads": 1,
            "linear_num_value_heads": 2,
            "linear_key_head_dim": 128,
            "linear_value_head_dim": 128,
            "linear_conv_kernel_dim": 4,
            "intermediate_size": 512,
            "max_position_embeddings": 4096,
            "vocab_size": 32
        })
    }

    #[test]
    fn authenticated_source_projection_matches_converter_defaults() {
        let projected = qwen35_config_from_authenticated_source(&valid_config()).unwrap();
        assert_eq!(projected.variant, Qwen35Variant::Dense);
        assert_eq!(projected.full_attention_interval, 4);
        assert_eq!(projected.rotary_dim, 16);
        assert_eq!(projected.mrope_section, [11, 11, 10, 0]);
        assert_eq!(projected.intermediate_size, Some(512));
        assert!(projected.moe.is_none());
    }

    #[test]
    fn authenticated_source_projection_rejects_runtime_geometry_drift() {
        let mut config = valid_config();
        config["linear_value_head_dim"] = json!(64);
        assert!(qwen35_config_from_authenticated_source(&config).is_err());
    }

    #[test]
    fn authenticated_source_projection_rejects_malformed_optional_and_unbounded_geometry() {
        let base = valid_config();
        for (key, malformed) in [
            ("num_key_value_heads", json!("1")),
            ("head_dim", json!("64")),
            ("full_attention_interval", json!("4")),
            ("mtp_num_hidden_layers", json!("1")),
            ("mtp_use_dedicated_embeddings", json!("false")),
            ("attn_output_gate", json!("true")),
        ] {
            let mut config = base.clone();
            config[key] = malformed;
            assert!(
                qwen35_config_from_authenticated_source(&config).is_err(),
                "malformed present field {key} must reject"
            );
        }

        let mut zero_heads = base.clone();
        zero_heads["linear_num_key_heads"] = json!(0);
        assert!(qwen35_config_from_authenticated_source(&zero_heads).is_err());

        let mut huge_layers = base;
        huge_layers["num_hidden_layers"] = json!(1_000_000_000_u64);
        assert!(qwen35_config_from_authenticated_source(&huge_layers).is_err());
    }
}
