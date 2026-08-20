use super::sse::CompletedResponse;
use super::wire::Message;

/// Session-only conversation history. A turn is transactional: neither the
/// user message nor its assistant response is retained until the stream ends
/// successfully.
#[derive(Debug)]
pub(crate) struct Transcript {
    system: Option<String>,
    messages: Vec<Message>,
}

impl Transcript {
    pub(crate) fn new(system: Option<String>) -> Self {
        let mut transcript = Self {
            system,
            messages: Vec::new(),
        };
        transcript.reset();
        transcript
    }

    pub(crate) fn pending(&self, user: &str) -> Vec<Message> {
        let mut pending = self.messages.clone();
        pending.push(Message::text("user", user));
        pending
    }

    pub(crate) fn commit(&mut self, user: &str, response: &CompletedResponse) {
        self.messages.push(Message::text("user", user));
        self.messages.push(Message {
            role: "assistant".to_owned(),
            content: (!response.content.is_empty()).then(|| response.content.clone()),
            reasoning_content: (!response.reasoning_content.is_empty())
                .then(|| response.reasoning_content.clone()),
            tool_calls: (!response.tool_calls.is_empty()).then(|| response.tool_calls.clone()),
        });
    }

    pub(crate) fn reset(&mut self) {
        self.messages.clear();
        if let Some(system) = &self.system {
            self.messages.push(Message::text("system", system));
        }
    }

    #[cfg(test)]
    pub(crate) fn messages(&self) -> &[Message] {
        &self.messages
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chat::wire::{Timing, ToolCall, ToolFunction, Usage};

    fn completed() -> CompletedResponse {
        CompletedResponse {
            content: "answer".into(),
            reasoning_content: "thought".into(),
            tool_calls: vec![ToolCall {
                id: "call_1".into(),
                call_type: "function".into(),
                function: ToolFunction {
                    name: "inspect".into(),
                    arguments: "{\"path\":\"src\"}".into(),
                },
            }],
            finish_reason: "stop".into(),
            usage: Some(Usage::default()),
            timing: Some(Timing::default()),
        }
    }

    #[test]
    fn only_successful_turns_enter_history() {
        let mut transcript = Transcript::new(Some("diagnose".into()));
        let pending = transcript.pending("first");
        assert_eq!(pending.len(), 2);
        assert_eq!(transcript.messages().len(), 1);

        transcript.commit("first", &completed());
        let second = transcript.pending("second");
        assert_eq!(second.len(), 4);
        assert_eq!(second[1].content.as_deref(), Some("first"));
        assert_eq!(second[2].reasoning_content.as_deref(), Some("thought"));
        assert_eq!(second[2].tool_calls.as_ref().unwrap()[0].id, "call_1");
        assert_eq!(second[3].content.as_deref(), Some("second"));
    }
}
