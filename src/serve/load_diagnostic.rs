//! Public diagnostics for model-load failures.
//!
//! Loader errors may contain operator paths, credentials embedded in URLs, or
//! other internal context. HTTP responses therefore expose only typed,
//! allow-listed details while the complete causal chain remains in server
//! logs.

use std::fmt;

use crate::inference::glp::GlpLoadError;
use crate::serve::multi_model::HotSwapError;

const MAX_PUBLIC_TENSOR_NAME_BYTES: usize = 160;

/// Typed cause emitted when the native loader requires a tensor that the
/// selected GGUF does not contain.
#[derive(Debug)]
pub(crate) struct MissingGgufTensor {
    tensor: String,
}

impl MissingGgufTensor {
    pub(crate) fn new(tensor: impl Into<String>) -> Self {
        Self {
            tensor: tensor.into(),
        }
    }

    fn public_tensor(&self) -> Option<&str> {
        let tensor = self.tensor.as_str();
        (!tensor.is_empty()
            && tensor.len() <= MAX_PUBLIC_TENSOR_NAME_BYTES
            && tensor
                .bytes()
                .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'.' | b'_' | b'-')))
        .then_some(tensor)
    }
}

impl fmt::Display for MissingGgufTensor {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "required tensor '{}' is missing from GGUF",
            self.tensor
        )
    }
}

impl std::error::Error for MissingGgufTensor {}

/// A detail that is safe to reflect across the authenticated HTTP boundary.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PublicLoadDiagnostic {
    MissingRequiredTensor(String),
    /// ADR-053: the server's `--glp` vector cannot be used with this model.
    GlpUnsupportedArchitecture(String),
    GlpUnusable,
    LoaderRejected,
    FileMetadataUnavailable,
}

impl fmt::Display for PublicLoadDiagnostic {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::MissingRequiredTensor(tensor) => write!(
                formatter,
                "selected GGUF is missing required tensor '{tensor}'"
            ),
            Self::GlpUnsupportedArchitecture(architecture) => write!(
                formatter,
                "this server applies a GLP steering vector (--glp), and `{architecture}` models \
                 do not support GLP; serve this model from a server started without --glp"
            ),
            Self::GlpUnusable => write!(
                formatter,
                "this server's GLP steering vector (--glp) cannot be applied to the selected \
                 model, for example because it was made for a different checkpoint; inspect \
                 server diagnostics"
            ),
            Self::LoaderRejected => write!(
                formatter,
                "model loader rejected selected GGUF; inspect server diagnostics"
            ),
            Self::FileMetadataUnavailable => write!(
                formatter,
                "cannot inspect selected GGUF file metadata; inspect server diagnostics"
            ),
        }
    }
}

pub(crate) fn public_hotswap_diagnostic(error: &HotSwapError) -> PublicLoadDiagnostic {
    match error {
        HotSwapError::LoaderFailed(error) => error
            .chain()
            .find_map(|cause| cause.downcast_ref::<GlpLoadError>())
            .map(public_glp_diagnostic)
            .or_else(|| {
                error
                    .chain()
                    .find_map(|cause| cause.downcast_ref::<MissingGgufTensor>())
                    .and_then(MissingGgufTensor::public_tensor)
                    .map(|tensor| PublicLoadDiagnostic::MissingRequiredTensor(tensor.to_owned()))
            })
            .unwrap_or(PublicLoadDiagnostic::LoaderRejected),
        HotSwapError::FileSize { .. } => PublicLoadDiagnostic::FileMetadataUnavailable,
        HotSwapError::PoolRefused(_) => PublicLoadDiagnostic::LoaderRejected,
    }
}

fn public_glp_diagnostic(error: &GlpLoadError) -> PublicLoadDiagnostic {
    match error {
        GlpLoadError::UnsupportedArchitecture { architecture, .. }
            if !architecture.is_empty()
                && architecture.len() <= 64
                && architecture
                    .bytes()
                    .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'_' | b'-')) =>
        {
            PublicLoadDiagnostic::GlpUnsupportedArchitecture(architecture.clone())
        }
        _ => PublicLoadDiagnostic::GlpUnusable,
    }
}

/// Full diagnostic for local server logs only.
pub(crate) fn private_hotswap_diagnostic(error: &HotSwapError) -> String {
    match error {
        HotSwapError::LoaderFailed(error) => format!("{error:#}"),
        HotSwapError::FileSize { path, source } => {
            format!("cannot read GGUF size at {}: {source}", path.display())
        }
        HotSwapError::PoolRefused(error) => error.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn glp_refusals_are_named_without_leaking_references_or_paths() {
        let unusable = HotSwapError::LoaderFailed(
            anyhow::Error::new(GlpLoadError::Unusable(anyhow::anyhow!(
                "GLP base repository mismatch: https://huggingface.co/private/vector?token=hf_secret"
            )))
            .context("/private/operator/model.gguf"),
        );
        let public = public_hotswap_diagnostic(&unusable);
        assert_eq!(public, PublicLoadDiagnostic::GlpUnusable);
        let text = public.to_string();
        assert!(text.contains("GLP steering vector (--glp)"), "{text}");
        assert!(!text.contains("hf_secret") && !text.contains("/private/"), "{text}");
        assert!(private_hotswap_diagnostic(&unusable).contains("repository mismatch"));

        let unsupported = HotSwapError::LoaderFailed(anyhow::Error::new(
            GlpLoadError::UnsupportedArchitecture {
                architecture: "gemma4".into(),
                model_path: "/private/operator/gemma.gguf".into(),
            },
        ));
        let public = public_hotswap_diagnostic(&unsupported);
        assert_eq!(
            public,
            PublicLoadDiagnostic::GlpUnsupportedArchitecture("gemma4".into())
        );
        assert!(!public.to_string().contains("/private/"));

        let hostile = HotSwapError::LoaderFailed(anyhow::Error::new(
            GlpLoadError::UnsupportedArchitecture {
                architecture: "gemma4\n/private/x".into(),
                model_path: "/m.gguf".into(),
            },
        ));
        assert_eq!(
            public_hotswap_diagnostic(&hostile),
            PublicLoadDiagnostic::GlpUnusable
        );
    }

    #[test]
    fn typed_missing_tensor_is_actionable_but_arbitrary_context_is_redacted() {
        let error = anyhow::Error::new(MissingGgufTensor::new("output.weight"))
            .context("credential hf_secret and /private/operator/model.gguf")
            .context("Qwen35Model::load_from_gguf");
        let diagnostic = public_hotswap_diagnostic(&HotSwapError::LoaderFailed(error));
        assert_eq!(
            diagnostic.to_string(),
            "selected GGUF is missing required tensor 'output.weight'"
        );
        assert!(!diagnostic.to_string().contains("hf_secret"));
        assert!(!diagnostic.to_string().contains("/private"));
    }

    #[test]
    fn unknown_loader_error_is_generic() {
        let error = anyhow::anyhow!("token hf_secret at /private/operator/model.gguf");
        assert_eq!(
            public_hotswap_diagnostic(&HotSwapError::LoaderFailed(error)),
            PublicLoadDiagnostic::LoaderRejected
        );
    }

    #[test]
    fn hostile_tensor_name_is_not_reflected() {
        let error = anyhow::Error::new(MissingGgufTensor::new("output.weight/hf_secret"));
        assert_eq!(
            public_hotswap_diagnostic(&HotSwapError::LoaderFailed(error)),
            PublicLoadDiagnostic::LoaderRejected
        );
    }
}
