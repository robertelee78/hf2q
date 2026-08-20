//! Shared, byte-bound Hugging Face chat-render input resolution.
//!
//! Conversion and calibration pass the exact `tokenizer.json` bytes they
//! parsed and receive the same tokenizer-config value, selected template, and
//! input-bundle hash. This prevents either path from silently falling back
//! after the other rejected malformed source bytes.

use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use thiserror::Error;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ChatTemplateSource {
    Sidecar,
    TokenizerConfig,
    FamilyFallback,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ResolvedChatTemplate {
    pub template: String,
    pub source: ChatTemplateSource,
    pub sha256: String,
}

#[derive(Debug, Clone)]
pub struct ResolvedChatRenderInputs {
    pub tokenizer_config: Option<serde_json::Value>,
    /// Exact optional bytes parsed above; callers can bind them to an opaque
    /// verified-source manifest without reopening either path.
    pub tokenizer_config_bytes: Option<Vec<u8>>,
    pub chat_template_sidecar_bytes: Option<Vec<u8>>,
    pub template: Option<ResolvedChatTemplate>,
    pub tokenizer_json_sha256: String,
    /// Hash of the exact tokenizer.json, tokenizer_config.json, sidecar, and
    /// selected/fallback template inputs that can affect rendering.
    pub tokenizer_bundle_sha256: String,
}

#[derive(Debug, Error)]
pub enum ChatTemplateResolveError {
    #[error("read tokenizer config {path}: {source}")]
    ReadTokenizerConfig {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("parse tokenizer config {path}: {detail}")]
    ParseTokenizerConfig { path: PathBuf, detail: String },
    #[error("read chat template sidecar {path}: {source}")]
    ReadSidecar {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("chat template sidecar {path} is not UTF-8")]
    SidecarNotUtf8 { path: PathBuf },
    #[error("tokenizer_config.json chat_template must be a string when present")]
    InvalidTokenizerConfigTemplate,
    #[error(
        "named tokenizer_config.json chat_template sets are not yet representable in hf2q GGUF metadata"
    )]
    NamedTemplateSetUnsupported,
}

fn sha256_bytes(bytes: &[u8]) -> String {
    hex::encode(Sha256::digest(bytes))
}

fn frame_optional(hasher: &mut Sha256, name: &str, bytes: Option<&[u8]>) {
    hasher.update((name.len() as u64).to_le_bytes());
    hasher.update(name.as_bytes());
    match bytes {
        Some(bytes) => {
            hasher.update([1]);
            hasher.update((bytes.len() as u64).to_le_bytes());
            hasher.update(bytes);
        }
        None => hasher.update([0]),
    }
}

/// Resolve every tokenizer/template input from one source snapshot.
///
/// Hugging Face also permits a dictionary of named templates. hf2q's current
/// GGUF/runtime contract has one `tokenizer.chat_template` string, so v1
/// rejects named sets identically in conversion and calibration instead of
/// selecting different implicit defaults.
pub fn resolve_chat_render_inputs(
    model_dir: &Path,
    tokenizer_json_bytes: &[u8],
    arch: &str,
) -> Result<ResolvedChatRenderInputs, ChatTemplateResolveError> {
    let config_path = model_dir.join("tokenizer_config.json");
    let config_bytes = match std::fs::read(&config_path) {
        Ok(bytes) => Some(bytes),
        Err(source) if source.kind() == std::io::ErrorKind::NotFound => None,
        Err(source) => {
            return Err(ChatTemplateResolveError::ReadTokenizerConfig {
                path: config_path.clone(),
                source,
            })
        }
    };
    let tokenizer_config: Option<serde_json::Value> = config_bytes
        .as_ref()
        .map(|bytes| {
            serde_json::from_slice(bytes).map_err(|error| {
                ChatTemplateResolveError::ParseTokenizerConfig {
                    path: config_path.clone(),
                    detail: error.to_string(),
                }
            })
        })
        .transpose()?;

    let sidecar_path = model_dir.join("chat_template.jinja");
    let sidecar_bytes = match std::fs::read(&sidecar_path) {
        Ok(bytes) => Some(bytes),
        Err(source) if source.kind() == std::io::ErrorKind::NotFound => None,
        Err(source) => {
            return Err(ChatTemplateResolveError::ReadSidecar {
                path: sidecar_path.clone(),
                source,
            })
        }
    };
    let (template, source) = if let Some(bytes) = sidecar_bytes.as_ref() {
        let template = std::str::from_utf8(bytes)
            .map_err(|_| ChatTemplateResolveError::SidecarNotUtf8 {
                path: sidecar_path.clone(),
            })?
            .to_owned();
        (Some(template), Some(ChatTemplateSource::Sidecar))
    } else if let Some(value) = tokenizer_config
        .as_ref()
        .and_then(|config| config.get("chat_template"))
    {
        match value {
            serde_json::Value::String(template) => (
                Some(template.clone()),
                Some(ChatTemplateSource::TokenizerConfig),
            ),
            serde_json::Value::Object(_) => {
                return Err(ChatTemplateResolveError::NamedTemplateSetUnsupported)
            }
            _ => return Err(ChatTemplateResolveError::InvalidTokenizerConfigTemplate),
        }
    } else if let Some(template) = super::chat_templates::arch_default_chat_template(arch) {
        (
            Some(template.to_owned()),
            Some(ChatTemplateSource::FamilyFallback),
        )
    } else {
        (None, None)
    };
    let resolved = template.map(|template| ResolvedChatTemplate {
        sha256: sha256_bytes(template.as_bytes()),
        template,
        source: source.expect("template source exists with template"),
    });

    let mut bundle = Sha256::new();
    bundle.update(b"hf2q-chat-render-inputs-v1");
    frame_optional(&mut bundle, "tokenizer.json", Some(tokenizer_json_bytes));
    frame_optional(
        &mut bundle,
        "tokenizer_config.json",
        config_bytes.as_deref(),
    );
    frame_optional(&mut bundle, "chat_template.jinja", sidecar_bytes.as_deref());
    frame_optional(&mut bundle, "architecture", Some(arch.as_bytes()));
    frame_optional(
        &mut bundle,
        "resolved_chat_template",
        resolved.as_ref().map(|value| value.template.as_bytes()),
    );

    Ok(ResolvedChatRenderInputs {
        tokenizer_config,
        tokenizer_config_bytes: config_bytes,
        chat_template_sidecar_bytes: sidecar_bytes,
        template: resolved,
        tokenizer_json_sha256: sha256_bytes(tokenizer_json_bytes),
        tokenizer_bundle_sha256: hex::encode(bundle.finalize()),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn priority_hash_and_bundle_are_exact() {
        let temp = tempfile::tempdir().unwrap();
        std::fs::write(
            temp.path().join("tokenizer_config.json"),
            br#"{"chat_template":"config-template"}"#,
        )
        .unwrap();
        let from_config = resolve_chat_render_inputs(temp.path(), b"tokenizer", "qwen35").unwrap();
        assert_eq!(
            from_config.template.as_ref().unwrap().source,
            ChatTemplateSource::TokenizerConfig
        );

        std::fs::write(temp.path().join("chat_template.jinja"), "sidecar-template").unwrap();
        let from_sidecar = resolve_chat_render_inputs(temp.path(), b"tokenizer", "qwen35").unwrap();
        assert_eq!(
            from_sidecar.template.as_ref().unwrap().source,
            ChatTemplateSource::Sidecar
        );
        assert_ne!(
            from_sidecar.template.as_ref().unwrap().sha256,
            from_config.template.as_ref().unwrap().sha256
        );
        assert_ne!(
            from_sidecar.tokenizer_bundle_sha256,
            from_config.tokenizer_bundle_sha256
        );
    }

    #[test]
    fn malformed_and_named_config_fail_closed() {
        let temp = tempfile::tempdir().unwrap();
        std::fs::write(temp.path().join("tokenizer_config.json"), b"{").unwrap();
        assert!(matches!(
            resolve_chat_render_inputs(temp.path(), b"tokenizer", "qwen35"),
            Err(ChatTemplateResolveError::ParseTokenizerConfig { .. })
        ));

        std::fs::write(
            temp.path().join("tokenizer_config.json"),
            br#"{"chat_template":{"default":"a","tool_use":"b"}}"#,
        )
        .unwrap();
        assert!(matches!(
            resolve_chat_render_inputs(temp.path(), b"tokenizer", "qwen35"),
            Err(ChatTemplateResolveError::NamedTemplateSetUnsupported)
        ));
    }
}
