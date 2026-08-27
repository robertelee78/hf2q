//! Bounded, structured startup telemetry shared by direct serve and owned chat.
//!
//! Owned chat keeps the server child's stderr private, so human-readable log
//! output cannot be its source of truth. These events cross a dedicated
//! inherited private datagram channel and are rendered by the parent after
//! strict decoding. The sibling stream remains the sole READY/DETACH authority.

use std::time::Duration;

use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum StartupOrigin {
    Hf2qConversion,
    Hf2qModelCache,
    ManualDownload,
    ManualStructuralMatch,
    HuggingFaceCache,
    HuggingFaceCacheStructuralMatch,
    HuggingFaceDownload,
    ExistingManagedFile,
    ManagedModel,
    OtherLocal,
}

impl StartupOrigin {
    pub(crate) fn from_internal(origin: &str) -> Self {
        match origin {
            "local_receipt" => Self::Hf2qConversion,
            "local_cache" | "cache_manifest" => Self::Hf2qModelCache,
            "manual_adoption" | "manual" | "local_adoption" => Self::ManualDownload,
            "manual_structural" | "manual_metadata" => Self::ManualStructuralMatch,
            "hf_hub_cache_adoption" => Self::HuggingFaceCache,
            "hf_hub_cache_structural" => Self::HuggingFaceCacheStructuralMatch,
            "hosted_download" => Self::HuggingFaceDownload,
            "existing_destination" => Self::ExistingManagedFile,
            "managed_binding" => Self::ManagedModel,
            _ => Self::OtherLocal,
        }
    }

    pub(crate) const fn label(self) -> &'static str {
        match self {
            Self::Hf2qConversion => "hf2q conversion",
            Self::Hf2qModelCache => "hf2q model cache",
            Self::ManualDownload => "manual download",
            Self::ManualStructuralMatch => "manual download, structural match",
            Self::HuggingFaceCache => "Hugging Face cache",
            Self::HuggingFaceCacheStructuralMatch => "Hugging Face cache, structural match",
            Self::HuggingFaceDownload => "Hugging Face download",
            Self::ExistingManagedFile => "existing managed file",
            Self::ManagedModel => "managed model",
            Self::OtherLocal => "local model store",
        }
    }
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum TextOnlyReason {
    ProjectorUnavailable,
    ProjectorPairRejected,
    ProjectorLoadRejected,
}

impl TextOnlyReason {
    pub(crate) const fn label(self) -> &'static str {
        match self {
            Self::ProjectorUnavailable => {
                "no unambiguous compatible multimodal projector was available"
            }
            Self::ProjectorPairRejected => {
                "the automatic multimodal projector pair failed safety checks"
            }
            Self::ProjectorLoadRejected => {
                "the automatic multimodal projector failed compatibility or warmup"
            }
        }
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(tag = "phase", rename_all = "snake_case", deny_unknown_fields)]
pub(crate) enum StartupEvent {
    LocalSearch {
        repository: String,
        requested_quant: Option<String>,
    },
    LocalCandidate {
        quant: String,
        origin: StartupOrigin,
        bytes: u64,
        filename: String,
    },
    VerifyStart {
        artifact: String,
        bytes: u64,
        filename: String,
    },
    VerifyProgress {
        artifact: String,
        completed_bytes: u64,
        total_bytes: u64,
        elapsed_ms: u64,
    },
    LocalReady {
        quant: String,
        origin: StartupOrigin,
        filename: String,
    },
    ModelPrepared {
        quant: String,
        origin: StartupOrigin,
        filename: String,
    },
    HubMetadata {
        repository: String,
    },
    HostedDownload {
        filename: String,
        bytes: u64,
    },
    HostedDownloadProgress {
        filename: String,
        completed_bytes: u64,
        total_bytes: u64,
        bytes_per_second: Option<u64>,
        elapsed_ms: u64,
    },
    ProjectorPrepare {
        filename: String,
        bytes: u64,
    },
    NativeConversion {
        repository: String,
        quant: String,
    },
    TextLoad {
        quant: String,
        bytes: u64,
        filename: String,
    },
    TextReady {
        elapsed_ms: u64,
    },
    ProjectorLoad {
        bytes: u64,
        filename: String,
    },
    ProjectorReady {
        elapsed_ms: u64,
    },
    TextOnlyFallback {
        reason: TextOnlyReason,
    },
}

impl StartupEvent {
    pub(crate) fn wire_valid(&self) -> bool {
        match self {
            Self::LocalSearch {
                repository,
                requested_quant,
            } => valid_repository(repository) && requested_quant.as_deref().is_none_or(valid_quant),
            Self::LocalCandidate {
                quant,
                origin,
                bytes,
                filename,
            } => {
                let _ = origin;
                valid_quant(quant) && *bytes > 0 && valid_filename(filename)
            }
            Self::VerifyStart {
                artifact,
                bytes,
                filename,
            } => valid_artifact_label(artifact) && *bytes > 0 && valid_filename(filename),
            Self::VerifyProgress {
                artifact,
                completed_bytes,
                total_bytes,
                ..
            } => {
                valid_artifact_label(artifact)
                    && *total_bytes > 0
                    && *completed_bytes <= *total_bytes
            }
            Self::LocalReady {
                quant,
                origin,
                filename,
            }
            | Self::ModelPrepared {
                quant,
                origin,
                filename,
            } => {
                let _ = origin;
                valid_quant(quant) && valid_filename(filename)
            }
            Self::HubMetadata { repository } => valid_repository(repository),
            Self::HostedDownload { filename, bytes }
            | Self::ProjectorPrepare { filename, bytes } => valid_filename(filename) && *bytes > 0,
            Self::HostedDownloadProgress {
                filename,
                completed_bytes,
                total_bytes,
                bytes_per_second,
                ..
            } => {
                valid_filename(filename)
                    && *total_bytes > 0
                    && *completed_bytes <= *total_bytes
                    && bytes_per_second.is_none_or(|rate| rate > 0)
            }
            Self::NativeConversion { repository, quant } => {
                valid_repository(repository) && valid_quant(quant)
            }
            Self::TextLoad {
                quant,
                bytes,
                filename,
            } => valid_quant(quant) && *bytes > 0 && valid_filename(filename),
            Self::TextReady { .. } | Self::ProjectorReady { .. } => true,
            Self::ProjectorLoad { bytes, filename } => *bytes > 0 && valid_filename(filename),
            Self::TextOnlyFallback { .. } => true,
        }
    }

    pub(crate) fn render(&self) -> String {
        match self {
            Self::LocalSearch {
                repository,
                requested_quant,
            } => format!(
                "local: searching managed, converted, manual, and Hugging Face cache artifacts for {}{}",
                clean(repository),
                requested_quant
                    .as_deref()
                    .map(|quant| format!(":{}", clean(quant)))
                    .unwrap_or_default()
            ),
            Self::LocalCandidate {
                quant,
                origin,
                bytes,
                filename,
            } => format!(
                "local: found {} `{}` ({}, {}); inspecting bounded GGUF metadata before reuse",
                clean(quant),
                clean(filename),
                human_bytes(*bytes),
                origin.label()
            ),
            Self::VerifyStart {
                artifact,
                bytes,
                filename,
            } => format!(
                "verify: checking local {} `{}` ({})",
                clean(artifact),
                clean(filename),
                human_bytes(*bytes)
            ),
            Self::VerifyProgress {
                artifact,
                completed_bytes,
                total_bytes,
                elapsed_ms,
            } => {
                let percent = if *total_bytes == 0 {
                    100
                } else {
                    completed_bytes.saturating_mul(100) / total_bytes
                };
                let elapsed = Duration::from_millis(*elapsed_ms);
                let rate = if elapsed.is_zero() {
                    0.0
                } else {
                    *completed_bytes as f64 / elapsed.as_secs_f64()
                };
                let remaining = total_bytes.saturating_sub(*completed_bytes);
                let eta = if rate > 0.0 {
                    format_duration(Duration::from_secs_f64(remaining as f64 / rate))
                } else {
                    "unknown".to_owned()
                };
                format!(
                    "verify: local {} {}/{} ({}%, {}/s, ETA {})",
                    clean(artifact),
                    human_bytes(*completed_bytes),
                    human_bytes(*total_bytes),
                    percent.min(100),
                    human_bytes(rate.max(0.0) as u64),
                    eta
                )
            }
            Self::LocalReady {
                quant,
                origin,
                filename,
            } => format!(
                "local: admitted compatible {} `{}` ({}); no model download or full-file hash needed",
                clean(quant),
                clean(filename),
                origin.label()
            ),
            Self::ModelPrepared {
                quant,
                origin,
                filename,
            } => format!(
                "prepare: {} `{}` is ready in the managed model store ({})",
                clean(quant),
                clean(filename),
                origin.label()
            ),
            Self::HubMetadata { repository } => format!(
                "hub: querying exact-revision metadata for {} (no payload transfer)",
                clean(repository)
            ),
            Self::HostedDownload {
                filename,
                bytes,
            } => format!(
                "download: fetching {} ({}) into the Hugging Face cache; final artifact is a managed-store symlink",
                clean(filename),
                human_bytes(*bytes)
            ),
            Self::HostedDownloadProgress {
                filename,
                completed_bytes,
                total_bytes,
                bytes_per_second,
                elapsed_ms,
            } => {
                let percent = completed_bytes.saturating_mul(100) / total_bytes;
                let measured_rate = if *elapsed_ms == 0 {
                    0
                } else {
                    ((*completed_bytes as u128).saturating_mul(1_000) / u128::from(*elapsed_ms))
                        .min(u128::from(u64::MAX)) as u64
                };
                let rate = bytes_per_second.unwrap_or(measured_rate);
                let remaining = total_bytes.saturating_sub(*completed_bytes);
                let eta = if rate > 0 {
                    format_duration(Duration::from_secs_f64(remaining as f64 / rate as f64))
                } else {
                    "unknown".to_owned()
                };
                format!(
                    "download: {} {}/{} ({}%, {}/s, ETA {})",
                    clean(filename),
                    human_bytes(*completed_bytes),
                    human_bytes(*total_bytes),
                    percent.min(100),
                    human_bytes(rate),
                    eta
                )
            }
            Self::ProjectorPrepare { filename, bytes } => format!(
                "projector: locating or downloading exact-revision {} ({})",
                clean(filename),
                human_bytes(*bytes)
            ),
            Self::NativeConversion {
                repository,
                quant,
            } => format!(
                "convert: no compatible hosted GGUF; checking/downloading missing source weights for {}, then converting to {} in the managed model store",
                clean(repository),
                clean(quant)
            ),
            Self::TextLoad {
                quant,
                bytes,
                filename,
            } => format!(
                "load: loading local {} GGUF `{}` ({}) into Metal and warming kernels",
                clean(quant),
                clean(filename),
                human_bytes(*bytes)
            ),
            Self::TextReady { elapsed_ms } => {
                format!(
                    "load: text model load and warmup complete in {}",
                    format_millis(*elapsed_ms)
                )
            }
            Self::ProjectorLoad { bytes, filename } => format!(
                "load: loading and warming local multimodal projector `{}` ({})",
                clean(filename),
                human_bytes(*bytes)
            ),
            Self::ProjectorReady { elapsed_ms } => format!(
                "load: multimodal projector ready in {}",
                format_millis(*elapsed_ms)
            ),
            Self::TextOnlyFallback { reason } => {
                format!("warning: serving text-only because {}", reason.label())
            }
        }
    }

    pub(crate) fn heartbeat_label(&self) -> String {
        match self {
            Self::LocalSearch { .. } => "searching local model stores".into(),
            Self::LocalCandidate { .. } => {
                "inspecting bounded GGUF metadata and tensor directory".into()
            }
            Self::VerifyStart { .. } | Self::VerifyProgress { .. } => {
                "verifying immutable payload bytes".into()
            }
            Self::LocalReady { .. } | Self::ModelPrepared { .. } => {
                "preparing the compatible local model".into()
            }
            Self::HubMetadata { .. } => "querying Hugging Face metadata".into(),
            Self::HostedDownload { .. } | Self::HostedDownloadProgress { .. } => {
                "downloading the selected hosted GGUF".into()
            }
            Self::ProjectorPrepare { .. } => {
                "locating or downloading the multimodal projector".into()
            }
            Self::NativeConversion { .. } => "running native hf2q conversion".into(),
            Self::TextLoad { .. } => "loading and warming the text model".into(),
            Self::TextReady { .. } => "finishing server startup".into(),
            Self::ProjectorLoad { .. } => "loading and warming the multimodal projector".into(),
            Self::ProjectorReady { .. } => "finishing server startup".into(),
            Self::TextOnlyFallback { .. } => "finishing text-only server startup".into(),
        }
    }
}

fn valid_repository(value: &str) -> bool {
    let mut parts = value.split('/');
    let owner = parts.next().unwrap_or_default();
    let repository = parts.next().unwrap_or_default();
    value.len() <= 256
        && !owner.is_empty()
        && !repository.is_empty()
        && parts.next().is_none()
        && !value.chars().any(unsafe_display_char)
        && value
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || b"-._/".contains(&byte))
}

fn valid_quant(value: &str) -> bool {
    matches!(
        value,
        "Q2_K" | "Q3_K_M" | "Q4_K_M" | "Q5_K_M" | "Q6_K" | "Q8_0"
    )
}

fn valid_artifact_label(value: &str) -> bool {
    matches!(value, "text GGUF" | "multimodal projector")
}

fn valid_filename(value: &str) -> bool {
    !value.is_empty()
        && value.len() <= 255
        && !value.contains('/')
        && !value.contains('\\')
        && !value.chars().any(unsafe_display_char)
}

pub(crate) fn render_verified_ready(address: &str) -> String {
    format!("ready: verified hf2q endpoint {}", clean(address))
}

/// Bound untrusted operator-facing text and neutralize terminal controls,
/// including Unicode bidi overrides. Shared by every live terminal surface.
pub(crate) fn terminal_safe_text(value: &str) -> String {
    value
        .chars()
        .take(768)
        .map(|character| {
            if unsafe_display_char(character) {
                '\u{fffd}'
            } else {
                character
            }
        })
        .collect()
}

fn clean(value: &str) -> String {
    terminal_safe_text(value)
}

pub(crate) fn unsafe_display_char(character: char) -> bool {
    character.is_control()
        || matches!(
            character,
            '\u{061c}'
                | '\u{200e}'
                | '\u{200f}'
                | '\u{202a}'..='\u{202e}'
                | '\u{2066}'..='\u{2069}'
        )
}

pub(crate) fn human_bytes(bytes: u64) -> String {
    const KIB: f64 = 1024.0;
    const MIB: f64 = KIB * 1024.0;
    const GIB: f64 = MIB * 1024.0;
    let bytes = bytes as f64;
    if bytes >= GIB {
        format!("{:.1} GiB", bytes / GIB)
    } else if bytes >= MIB {
        format!("{:.1} MiB", bytes / MIB)
    } else if bytes >= KIB {
        format!("{:.1} KiB", bytes / KIB)
    } else {
        format!("{} B", bytes as u64)
    }
}

fn format_millis(milliseconds: u64) -> String {
    format_duration(Duration::from_millis(milliseconds))
}

pub(crate) fn format_duration(duration: Duration) -> String {
    let seconds = duration.as_secs();
    if seconds >= 3600 {
        format!("{}h {:02}m", seconds / 3600, (seconds % 3600) / 60)
    } else if seconds >= 60 {
        format!("{}m {:02}s", seconds / 60, seconds % 60)
    } else if seconds > 0 {
        format!("{}s", seconds)
    } else {
        format!("{:.1}s", duration.as_secs_f64())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn local_ready_explicitly_rules_out_a_download_and_full_file_hash() {
        let event = StartupEvent::LocalReady {
            quant: "Q6_K".into(),
            origin: StartupOrigin::ManualDownload,
            filename: "qwen.gguf".into(),
        };
        let rendered = event.render();
        assert!(rendered.contains("no model download or full-file hash needed"));
        assert!(rendered.contains("Q6_K"));
        assert!(rendered.contains("manual download"));
        assert!(!rendered.contains("manual_adoption"));
    }

    #[test]
    fn rendered_status_replaces_terminal_controls() {
        let event = StartupEvent::HubMetadata {
            repository: "owner/model\nsecret\u{202e}".into(),
        };
        let rendered = event.render();
        assert!(!rendered.contains('\n'));
        assert!(!rendered.contains('\u{202e}'));
        assert!(rendered.contains('\u{fffd}'));
    }

    #[test]
    fn verification_progress_includes_rate_and_eta() {
        let event = StartupEvent::VerifyProgress {
            artifact: "text GGUF".into(),
            completed_bytes: 5 * 1024 * 1024 * 1024,
            total_bytes: 10 * 1024 * 1024 * 1024,
            elapsed_ms: 5_000,
        };
        let rendered = event.render();
        assert!(rendered.contains("50%"));
        assert!(rendered.contains("GiB/s"));
        assert!(rendered.contains("ETA 5s"));
    }

    #[test]
    fn hosted_download_progress_includes_bytes_percent_rate_and_eta() {
        let event = StartupEvent::HostedDownloadProgress {
            filename: "model-q4_k_m.gguf".into(),
            completed_bytes: 5 * 1024 * 1024 * 1024,
            total_bytes: 10 * 1024 * 1024 * 1024,
            bytes_per_second: Some(1024 * 1024 * 1024),
            elapsed_ms: 5_000,
        };
        assert!(event.wire_valid());
        let rendered = event.render();
        assert!(rendered.contains("model-q4_k_m.gguf"));
        assert!(rendered.contains("5.0 GiB/10.0 GiB"));
        assert!(rendered.contains("50%"));
        assert!(rendered.contains("1.0 GiB/s"));
        assert!(rendered.contains("ETA 5s"));
    }

    #[test]
    fn wire_validation_rejects_paths_and_impossible_progress() {
        assert!(!StartupEvent::LocalReady {
            quant: "Q6_K".into(),
            origin: StartupOrigin::ManualDownload,
            filename: "/private/model.gguf".into(),
        }
        .wire_valid());
        assert!(!StartupEvent::HubMetadata {
            repository: "owner/".into(),
        }
        .wire_valid());
        assert!(!StartupEvent::LocalReady {
            quant: "Q6_K".into(),
            origin: StartupOrigin::ManualDownload,
            filename: "safe\u{202e}fdp.exe".into(),
        }
        .wire_valid());
        assert!(!StartupEvent::VerifyProgress {
            artifact: "text GGUF".into(),
            completed_bytes: 2,
            total_bytes: 1,
            elapsed_ms: 1,
        }
        .wire_valid());
        assert!(!StartupEvent::HostedDownloadProgress {
            filename: "model.gguf".into(),
            completed_bytes: 2,
            total_bytes: 1,
            bytes_per_second: Some(1),
            elapsed_ms: 1,
        }
        .wire_valid());
        assert!(!StartupEvent::HostedDownloadProgress {
            filename: "../model.gguf".into(),
            completed_bytes: 1,
            total_bytes: 2,
            bytes_per_second: Some(1),
            elapsed_ms: 1,
        }
        .wire_valid());
    }

    #[test]
    fn text_only_fallback_is_bounded_visible_and_control_safe() {
        let event = StartupEvent::TextOnlyFallback {
            reason: TextOnlyReason::ProjectorLoadRejected,
        };
        assert!(event.wire_valid());
        assert!(event.render().contains("serving text-only"));
        let wire = serde_json::to_string(&event).unwrap();
        assert!(wire.contains("projector_load_rejected"));
        assert!(!wire.contains('/'));
    }

    #[test]
    fn text_ready_never_claims_that_a_projector_is_loading() {
        assert_eq!(
            StartupEvent::TextReady { elapsed_ms: 7 }.heartbeat_label(),
            "finishing server startup"
        );
    }
}
