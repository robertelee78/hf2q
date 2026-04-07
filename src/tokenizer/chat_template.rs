//! Chat template rendering via minijinja.
//!
//! Loads a Jinja2 chat template (from `tokenizer_config.json` or a standalone
//! `.jinja` file) and renders it with a conversation `messages` array.
//!
//! Template priority:
//! 1. CLI `--chat-template` file override
//! 2. `chat_template` field in `tokenizer_config.json`
//! 3. Standalone `chat_template.jinja` file in model directory
//! 4. Error

use std::path::Path;

use minijinja::{context, Environment};
use serde::Serialize;
use thiserror::Error;
use tracing::debug;

/// Errors from chat template operations.
#[derive(Error, Debug)]
pub enum ChatTemplateError {
    #[error("No chat template found. Provide one with --chat-template")]
    NotFound,

    #[error("Failed to read template file {path}: {reason}")]
    ReadError { path: String, reason: String },

    #[error("Failed to parse tokenizer_config.json: {reason}")]
    ConfigParseError { reason: String },

    #[error("Template rendering failed: {reason}")]
    RenderError { reason: String },
}

/// A single message in a conversation.
#[derive(Debug, Clone, Serialize)]
pub struct Message {
    /// The role of the message sender (e.g. "user", "model", "system").
    pub role: String,
    /// The text content of the message.
    pub content: String,
}

/// A loaded chat template ready for rendering.
pub struct ChatTemplate {
    /// The raw Jinja2 template string.
    template_source: String,
}

impl ChatTemplate {
    /// Load a chat template from a model directory.
    ///
    /// Tries `tokenizer_config.json`'s `chat_template` field first,
    /// then falls back to a standalone `chat_template.jinja` file.
    pub fn from_model_dir(model_dir: &Path) -> Result<Self, ChatTemplateError> {
        // Try tokenizer_config.json first
        let config_path = model_dir.join("tokenizer_config.json");
        if config_path.exists() {
            let content = std::fs::read_to_string(&config_path).map_err(|e| {
                ChatTemplateError::ReadError {
                    path: config_path.display().to_string(),
                    reason: e.to_string(),
                }
            })?;

            let config: serde_json::Value =
                serde_json::from_str(&content).map_err(|e| ChatTemplateError::ConfigParseError {
                    reason: e.to_string(),
                })?;

            if let Some(template_str) = config.get("chat_template").and_then(|v| v.as_str()) {
                debug!("Loaded chat template from tokenizer_config.json");
                return Ok(Self {
                    template_source: template_str.to_string(),
                });
            }
        }

        // Fallback: standalone chat_template.jinja file
        let jinja_path = model_dir.join("chat_template.jinja");
        if jinja_path.exists() {
            let template_str =
                std::fs::read_to_string(&jinja_path).map_err(|e| ChatTemplateError::ReadError {
                    path: jinja_path.display().to_string(),
                    reason: e.to_string(),
                })?;
            debug!("Loaded chat template from chat_template.jinja");
            return Ok(Self {
                template_source: template_str,
            });
        }

        Err(ChatTemplateError::NotFound)
    }

    /// Load a chat template from a specific file path (CLI override).
    pub fn from_file(path: &Path) -> Result<Self, ChatTemplateError> {
        let template_str =
            std::fs::read_to_string(path).map_err(|e| ChatTemplateError::ReadError {
                path: path.display().to_string(),
                reason: e.to_string(),
            })?;
        debug!(path = %path.display(), "Loaded chat template from file override");
        Ok(Self {
            template_source: template_str,
        })
    }

    /// Render the template with the given conversation messages.
    ///
    /// The template receives:
    /// - `messages`: array of `{role, content}` objects
    /// - `add_generation_prompt`: true (appends the model turn prefix)
    /// - `bos_token`, `eos_token`: standard special token strings
    pub fn render(
        &self,
        messages: &[Message],
        bos_token: &str,
        eos_token: &str,
    ) -> Result<String, ChatTemplateError> {
        let mut env = Environment::new();

        // Register the `raise_exception` function that HuggingFace templates use
        env.add_function("raise_exception", |msg: String| -> Result<String, _> {
            Err(minijinja::Error::new(
                minijinja::ErrorKind::InvalidOperation,
                msg,
            ))
        });

        env.add_template("chat", &self.template_source)
            .map_err(|e| ChatTemplateError::RenderError {
                reason: format!("Template parse error: {e}"),
            })?;

        let tmpl = env.get_template("chat").map_err(|e| ChatTemplateError::RenderError {
            reason: format!("Template lookup error: {e}"),
        })?;

        let rendered = tmpl
            .render(context! {
                messages => messages,
                add_generation_prompt => true,
                bos_token => bos_token,
                eos_token => eos_token,
            })
            .map_err(|e| ChatTemplateError::RenderError {
                reason: format!("Template render error: {e}"),
            })?;

        Ok(rendered)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_render_simple_template() {
        let template = ChatTemplate {
            template_source: concat!(
                "{% for message in messages %}",
                "<start_of_turn>{{ message.role }}\n",
                "{{ message.content }}<end_of_turn>\n",
                "{% endfor %}",
                "{% if add_generation_prompt %}",
                "<start_of_turn>model\n",
                "{% endif %}"
            )
            .to_string(),
        };

        let messages = vec![
            Message {
                role: "user".to_string(),
                content: "Hello, world!".to_string(),
            },
        ];

        let rendered = template.render(&messages, "<bos>", "<eos>").unwrap();
        assert!(rendered.contains("<start_of_turn>user"));
        assert!(rendered.contains("Hello, world!"));
        assert!(rendered.contains("<start_of_turn>model"));
    }

    #[test]
    fn test_render_with_bos_eos_tokens() {
        let template = ChatTemplate {
            template_source: "{{ bos_token }}{% for m in messages %}{{ m.content }}{% endfor %}{{ eos_token }}".to_string(),
        };

        let messages = vec![Message {
            role: "user".to_string(),
            content: "Hi".to_string(),
        }];

        let rendered = template.render(&messages, "<bos>", "<eos>").unwrap();
        assert_eq!(rendered, "<bos>Hi<eos>");
    }

    #[test]
    fn test_render_multi_turn() {
        let template = ChatTemplate {
            template_source: concat!(
                "{% for message in messages %}",
                "[{{ message.role }}]: {{ message.content }}\n",
                "{% endfor %}",
                "{% if add_generation_prompt %}[model]: {% endif %}"
            )
            .to_string(),
        };

        let messages = vec![
            Message {
                role: "system".to_string(),
                content: "You are helpful.".to_string(),
            },
            Message {
                role: "user".to_string(),
                content: "What is 2+2?".to_string(),
            },
        ];

        let rendered = template.render(&messages, "", "").unwrap();
        assert!(rendered.contains("[system]: You are helpful."));
        assert!(rendered.contains("[user]: What is 2+2?"));
        assert!(rendered.contains("[model]: "));
    }

    #[test]
    fn test_from_model_dir_not_found() {
        let tmp = tempfile::tempdir().unwrap();
        let result = ChatTemplate::from_model_dir(tmp.path());
        assert!(matches!(result, Err(ChatTemplateError::NotFound)));
    }

    #[test]
    fn test_from_model_dir_config_json() {
        let tmp = tempfile::tempdir().unwrap();
        let config = serde_json::json!({
            "chat_template": "Hello {{ messages[0].content }}"
        });
        std::fs::write(
            tmp.path().join("tokenizer_config.json"),
            serde_json::to_string(&config).unwrap(),
        )
        .unwrap();

        let tmpl = ChatTemplate::from_model_dir(tmp.path()).unwrap();
        let messages = vec![Message {
            role: "user".to_string(),
            content: "world".to_string(),
        }];
        let rendered = tmpl.render(&messages, "", "").unwrap();
        assert_eq!(rendered, "Hello world");
    }

    #[test]
    fn test_from_model_dir_jinja_file_fallback() {
        let tmp = tempfile::tempdir().unwrap();
        // Write tokenizer_config.json WITHOUT chat_template
        std::fs::write(
            tmp.path().join("tokenizer_config.json"),
            "{}",
        )
        .unwrap();
        // Write standalone jinja file
        std::fs::write(
            tmp.path().join("chat_template.jinja"),
            "standalone: {{ messages[0].content }}",
        )
        .unwrap();

        let tmpl = ChatTemplate::from_model_dir(tmp.path()).unwrap();
        let messages = vec![Message {
            role: "user".to_string(),
            content: "test".to_string(),
        }];
        let rendered = tmpl.render(&messages, "", "").unwrap();
        assert_eq!(rendered, "standalone: test");
    }

    #[test]
    fn test_from_file_override() {
        let tmp = tempfile::tempdir().unwrap();
        let file_path = tmp.path().join("custom.jinja");
        std::fs::write(&file_path, "custom: {{ messages[0].role }}").unwrap();

        let tmpl = ChatTemplate::from_file(&file_path).unwrap();
        let messages = vec![Message {
            role: "user".to_string(),
            content: "ignored".to_string(),
        }];
        let rendered = tmpl.render(&messages, "", "").unwrap();
        assert_eq!(rendered, "custom: user");
    }

    #[test]
    fn test_from_file_not_found() {
        let result = ChatTemplate::from_file(Path::new("/nonexistent/template.jinja"));
        assert!(matches!(result, Err(ChatTemplateError::ReadError { .. })));
    }

    #[test]
    fn test_render_invalid_template() {
        let template = ChatTemplate {
            template_source: "{% invalid syntax %}".to_string(),
        };
        let result = template.render(&[], "", "");
        assert!(matches!(result, Err(ChatTemplateError::RenderError { .. })));
    }
}
