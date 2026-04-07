//! Streaming JSON state machine for detecting and parsing tool calls
//! in model output.
//!
//! The parser processes tokens as they arrive from the model and detects
//! tool call patterns. It supports two formats:
//!
//! 1. **Delimited format**: `<tool_call>{"name":"fn","arguments":{...}}</tool_call>`
//!    Used by Gemma 4 and other models with explicit tool call markers.
//!
//! 2. **Raw JSON format**: A JSON object with `"name"` and `"arguments"` fields
//!    at the top level, without delimiters.
//!
//! The parser tracks brace/bracket depth, string state, and escape sequences
//! to correctly identify JSON boundaries even when tokens arrive one character
//! at a time.

use tracing::warn;

// ---------------------------------------------------------------------------
// Parser events
// ---------------------------------------------------------------------------

/// Events emitted by the tool call parser as it processes tokens.
#[derive(Debug, Clone, PartialEq)]
pub enum ToolParserEvent {
    /// Plain text content (not part of a tool call).
    ContentDelta(String),
    /// A new tool call has started at the given index.
    ToolCallStart { index: usize },
    /// Fragment of the function name for the tool call at the given index.
    NameDelta { index: usize, text: String },
    /// Fragment of the function arguments for the tool call at the given index.
    ArgumentsDelta { index: usize, text: String },
    /// The tool call at the given index is complete.
    ToolCallEnd { index: usize },
}

// ---------------------------------------------------------------------------
// Internal state
// ---------------------------------------------------------------------------

/// What region of a tool call JSON object we are currently inside.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum JsonParsePhase {
    /// We are scanning for the "name" key.
    SeekingName,
    /// We found `"name"` and are reading the value string.
    ReadingName,
    /// We finished name, looking for "arguments" key.
    SeekingArguments,
    /// We are reading the arguments value (arbitrary JSON).
    ReadingArguments,
}

/// Top-level parser state.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TopLevelState {
    /// Not inside any tool call. Emitting content.
    Idle,
    /// We are inside a `<tool_call>` delimited block or a detected JSON object.
    InToolCallJson,
    /// Consuming the `</tool_call>` closing tag after JSON parsing completes.
    ConsumingCloseTag,
}

/// Tracks whether we are inside a JSON string or not.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum StringState {
    Outside,
    Inside,
    /// The previous character was a backslash inside a string.
    Escaped,
}

// ---------------------------------------------------------------------------
// ToolCallParser
// ---------------------------------------------------------------------------

/// Streaming tool call parser.
///
/// Feed tokens one at a time via `feed()`. The parser returns a list of
/// `ToolParserEvent` values for each token. Call `finalize()` at the end
/// of generation to flush any in-progress state.
pub struct ToolCallParser {
    /// Whether tool call parsing is active.
    active: bool,
    /// If set, only tool calls to this function name are recognized.
    forced_function: Option<String>,

    /// Top-level state.
    state: TopLevelState,
    /// Buffer for detecting the `<tool_call>` opening tag.
    tag_buffer: String,
    /// Buffer for accumulating the JSON body of a tool call.
    json_buffer: String,
    /// Where we are inside the tool call JSON.
    json_phase: JsonParsePhase,
    /// Brace depth tracker for the outer JSON object.
    outer_brace_depth: i32,
    /// Brace depth tracker for the arguments value.
    args_brace_depth: i32,
    /// String state tracker (for the arguments JSON sub-parse).
    string_state: StringState,
    /// Current tool call index (0-based, increments per parallel call).
    current_index: usize,
    /// Total number of tool calls detected so far.
    tool_call_count: usize,
    /// Whether we've emitted the name for the current tool call.
    name_emitted: bool,
    /// Accumulated name for the current tool call.
    current_name: String,
    /// Whether a tool call was detected in this generation.
    has_tool_calls: bool,
    /// Content buffer for text before tool calls.
    content_buffer: String,
    /// Whether the current tool call started with a `<tool_call>` delimiter.
    was_delimited: bool,
    /// Buffer for consuming the `</tool_call>` closing tag.
    close_tag_buffer: String,
}

impl ToolCallParser {
    /// Create a new parser.
    ///
    /// If `active` is false, the parser passes all tokens through as content.
    /// If `forced_function` is set, only tool calls matching that function
    /// name are recognized.
    pub fn new(active: bool, forced_function: Option<String>) -> Self {
        Self {
            active,
            forced_function,
            state: TopLevelState::Idle,
            tag_buffer: String::new(),
            json_buffer: String::new(),
            json_phase: JsonParsePhase::SeekingName,
            outer_brace_depth: 0,
            args_brace_depth: 0,
            string_state: StringState::Outside,
            current_index: 0,
            tool_call_count: 0,
            name_emitted: false,
            current_name: String::new(),
            has_tool_calls: false,
            content_buffer: String::new(),
            was_delimited: false,
            close_tag_buffer: String::new(),
        }
    }

    /// Whether the parser detected any tool calls during this generation.
    pub fn has_tool_calls(&self) -> bool {
        self.has_tool_calls
    }

    /// Feed a token string to the parser and receive events.
    pub fn feed(&mut self, token: &str) -> Vec<ToolParserEvent> {
        if !self.active {
            if token.is_empty() {
                return Vec::new();
            }
            return vec![ToolParserEvent::ContentDelta(token.to_string())];
        }

        let mut events = Vec::new();

        for ch in token.chars() {
            match self.state {
                TopLevelState::Idle => {
                    self.process_idle_char(ch, &mut events);
                }
                TopLevelState::InToolCallJson => {
                    self.process_json_char(ch, &mut events);
                }
                TopLevelState::ConsumingCloseTag => {
                    self.process_close_tag_char(ch, &mut events);
                }
            }
        }

        events
    }

    /// Finalize the parser at end of generation.
    ///
    /// Flushes any partially accumulated content or incomplete tool calls.
    pub fn finalize(&mut self) -> Vec<ToolParserEvent> {
        let mut events = Vec::new();

        // Flush any pending tag buffer as content
        if !self.tag_buffer.is_empty() {
            let text = std::mem::take(&mut self.tag_buffer);
            events.push(ToolParserEvent::ContentDelta(text));
        }

        // Flush any pending content buffer
        if !self.content_buffer.is_empty() {
            let text = std::mem::take(&mut self.content_buffer);
            events.push(ToolParserEvent::ContentDelta(text));
        }

        // Flush any pending close tag buffer
        if !self.close_tag_buffer.is_empty() {
            // Don't emit closing tag as content; just discard it
            self.close_tag_buffer.clear();
            self.state = TopLevelState::Idle;
        }

        // If we were in the middle of a tool call, attempt best-effort completion
        if self.state == TopLevelState::InToolCallJson {
            if self.json_phase == JsonParsePhase::ReadingArguments {
                // Emit whatever arguments we have accumulated
                let args = std::mem::take(&mut self.json_buffer);
                if !args.is_empty() {
                    events.push(ToolParserEvent::ArgumentsDelta {
                        index: self.current_index,
                        text: args,
                    });
                }
                warn!(
                    index = self.current_index,
                    "Tool call finalized with incomplete arguments"
                );
                events.push(ToolParserEvent::ToolCallEnd {
                    index: self.current_index,
                });
            } else if self.name_emitted {
                // We had a name but never got arguments
                warn!(
                    index = self.current_index,
                    "Tool call finalized without arguments"
                );
                events.push(ToolParserEvent::ToolCallEnd {
                    index: self.current_index,
                });
            }
            self.state = TopLevelState::Idle;
        }

        events
    }

    /// Process a character in the Idle state.
    ///
    /// Detects `<tool_call>` delimiters or the start of a raw JSON object
    /// that might be a tool call.
    fn process_idle_char(&mut self, ch: char, events: &mut Vec<ToolParserEvent>) {
        // Check for `<tool_call>` tag
        if ch == '<' && self.tag_buffer.is_empty() {
            self.tag_buffer.push(ch);
            return;
        }

        if !self.tag_buffer.is_empty() {
            self.tag_buffer.push(ch);

            // Full match
            if self.tag_buffer == "<tool_call>" {
                self.tag_buffer.clear();
                self.was_delimited = true;
                self.begin_tool_call(events);
                return;
            }

            // Partial prefix still matches
            if "<tool_call>".starts_with(&self.tag_buffer) {
                return;
            }

            // Not a match -- flush the tag buffer as content
            let text = std::mem::take(&mut self.tag_buffer);
            events.push(ToolParserEvent::ContentDelta(text));
            return;
        }

        // Check for start of raw JSON tool call (starts with `{`)
        if ch == '{' {
            // This might be the start of a tool call JSON object.
            // We'll begin parsing it tentatively.
            self.was_delimited = false;
            self.begin_tool_call(events);
            // Re-process this `{` in the JSON state
            self.process_json_char(ch, events);
            return;
        }

        // Regular content character
        events.push(ToolParserEvent::ContentDelta(ch.to_string()));
    }

    /// Begin a new tool call -- set up state for JSON parsing.
    fn begin_tool_call(&mut self, events: &mut Vec<ToolParserEvent>) {
        self.state = TopLevelState::InToolCallJson;
        self.json_buffer.clear();
        self.json_phase = JsonParsePhase::SeekingName;
        self.outer_brace_depth = 0;
        self.args_brace_depth = 0;
        self.string_state = StringState::Outside;
        self.name_emitted = false;
        self.current_name.clear();
        self.current_index = self.tool_call_count;

        events.push(ToolParserEvent::ToolCallStart {
            index: self.current_index,
        });
    }

    /// Process a character inside a tool call JSON object.
    fn process_json_char(&mut self, ch: char, events: &mut Vec<ToolParserEvent>) {
        // Track the outer JSON object brace depth
        match self.json_phase {
            JsonParsePhase::SeekingName | JsonParsePhase::ReadingName |
            JsonParsePhase::SeekingArguments => {
                // Track braces at the outer level
                match ch {
                    '{' => self.outer_brace_depth += 1,
                    '}' => {
                        self.outer_brace_depth -= 1;
                        if self.outer_brace_depth <= 0 {
                            // Outer object closed without finding expected fields
                            self.finish_tool_call(events);
                            return;
                        }
                    }
                    _ => {}
                }

                // Accumulate into json_buffer for key detection
                self.json_buffer.push(ch);
                self.check_json_keys(events);
            }
            JsonParsePhase::ReadingArguments => {
                self.process_arguments_char(ch, events);
            }
        }
    }

    /// Check if the JSON buffer reveals `"name"` or `"arguments"` keys.
    fn check_json_keys(&mut self, events: &mut Vec<ToolParserEvent>) {
        match self.json_phase {
            JsonParsePhase::SeekingName => {
                // Look for `"name"` followed by `:` and then a `"` starting the value
                if let Some(name_start) = find_json_string_value(&self.json_buffer, "name") {
                    self.json_phase = JsonParsePhase::ReadingName;
                    // Extract whatever name chars are already in the buffer
                    let name_fragment: String = self.json_buffer[name_start..].chars()
                        .take_while(|&c| c != '"')
                        .collect();
                    if !name_fragment.is_empty() {
                        self.current_name.push_str(&name_fragment);
                    }
                    // Check if the name is already complete (closing quote found)
                    let after_start = &self.json_buffer[name_start..];
                    if let Some(end_quote) = after_start.find('"') {
                        let name = after_start[..end_quote].to_string();
                        self.current_name = name.clone();
                        self.emit_name(events);
                        self.json_phase = JsonParsePhase::SeekingArguments;
                        self.json_buffer.clear();
                    }
                }
            }
            JsonParsePhase::ReadingName => {
                // We're reading the name string character by character.
                // Check if the last char is the closing quote.
                let buf = &self.json_buffer;
                if buf.ends_with('"') {
                    // The name is complete. Extract it from current_name plus buffer.
                    // The buffer since we entered ReadingName has the trailing chars.
                    // Actually, let's just look for the closing quote in the full buffer.
                    // Re-parse: find the name value in the full accumulated buffer.
                    // Simpler: we accumulated chars into json_buffer, and the name started
                    // after we detected the key. Let's find the value again.

                    // Trim the closing quote
                    let last_char = self.json_buffer.pop(); // remove the "
                    debug_assert_eq!(last_char, Some('"'));

                    // The remaining json_buffer content since ReadingName started
                    // is part of the name. But current_name may already have some.
                    // Let's re-extract properly.
                    if let Some(name_start) = find_json_string_value(&self.json_buffer, "name") {
                        let name = self.json_buffer[name_start..].to_string();
                        self.current_name = name;
                    }
                    self.emit_name(events);
                    self.json_phase = JsonParsePhase::SeekingArguments;
                    self.json_buffer.clear();
                }
            }
            JsonParsePhase::SeekingArguments => {
                // Look for `"arguments"` key followed by `:` and then the value start
                if let Some(args_start) = find_json_value_start(&self.json_buffer, "arguments") {
                    let value_char = self.json_buffer.as_bytes()[args_start] as char;
                    self.json_phase = JsonParsePhase::ReadingArguments;
                    self.json_buffer.clear();
                    self.string_state = StringState::Outside;
                    self.args_brace_depth = 0;

                    // If the value starts with `{` or `[`, it was already counted
                    // in outer_brace_depth by process_json_char. Undo that because
                    // the arguments parser tracks its own depth independently.
                    if value_char == '{' || value_char == '[' {
                        self.outer_brace_depth -= 1;
                    }

                    // Re-feed the value start char through the arguments parser
                    self.process_arguments_char(value_char, events);
                }
            }
            _ => {}
        }
    }

    /// Emit the function name.
    fn emit_name(&mut self, events: &mut Vec<ToolParserEvent>) {
        if self.name_emitted {
            return;
        }

        // Check forced function filter
        if let Some(ref forced) = self.forced_function {
            if self.current_name != *forced {
                // Not the forced function -- abort this tool call, emit as content
                warn!(
                    name = self.current_name,
                    forced = forced.as_str(),
                    "Tool call name does not match forced function"
                );
                // We won't emit ToolCallEnd; instead mark it as not a real tool call
                self.state = TopLevelState::Idle;
                return;
            }
        }

        self.name_emitted = true;
        self.has_tool_calls = true;
        events.push(ToolParserEvent::NameDelta {
            index: self.current_index,
            text: self.current_name.clone(),
        });
    }

    /// Process a character that is part of the arguments value.
    fn process_arguments_char(&mut self, ch: char, events: &mut Vec<ToolParserEvent>) {
        // Handle string state tracking
        match self.string_state {
            StringState::Inside => {
                if ch == '\\' {
                    self.string_state = StringState::Escaped;
                } else if ch == '"' {
                    self.string_state = StringState::Outside;
                }
            }
            StringState::Escaped => {
                self.string_state = StringState::Inside;
            }
            StringState::Outside => {
                if ch == '"' {
                    self.string_state = StringState::Inside;
                } else if ch == '{' || ch == '[' {
                    self.args_brace_depth += 1;
                } else if ch == '}' || ch == ']' {
                    self.args_brace_depth -= 1;

                    if ch == '}' && self.args_brace_depth < 0 {
                        // This closing brace belongs to the outer object, not arguments.
                        // The arguments value is complete.
                        let args_text = std::mem::take(&mut self.json_buffer);
                        if !args_text.is_empty() {
                            events.push(ToolParserEvent::ArgumentsDelta {
                                index: self.current_index,
                                text: args_text,
                            });
                        }
                        self.finish_tool_call(events);
                        return;
                    }

                    if self.args_brace_depth == 0 {
                        // The arguments JSON object/array is complete.
                        self.json_buffer.push(ch);
                        let args_text = std::mem::take(&mut self.json_buffer);
                        events.push(ToolParserEvent::ArgumentsDelta {
                            index: self.current_index,
                            text: args_text,
                        });
                        // Continue scanning for the closing `}` of the outer object
                        self.json_phase = JsonParsePhase::SeekingArguments;
                        self.json_buffer.clear();
                        return;
                    }
                }
            }
        }

        self.json_buffer.push(ch);

        // Emit arguments deltas incrementally (every character)
        // For streaming, we emit accumulated fragments
        // We'll batch in small chunks for efficiency
        if self.json_buffer.len() >= 1 {
            let text = std::mem::take(&mut self.json_buffer);
            events.push(ToolParserEvent::ArgumentsDelta {
                index: self.current_index,
                text,
            });
        }
    }

    /// Complete a tool call and prepare for the next one.
    fn finish_tool_call(&mut self, events: &mut Vec<ToolParserEvent>) {
        if self.name_emitted {
            events.push(ToolParserEvent::ToolCallEnd {
                index: self.current_index,
            });
            self.tool_call_count += 1;
        }

        // If the call was delimited with <tool_call>, consume the closing </tool_call> tag
        if self.was_delimited {
            self.state = TopLevelState::ConsumingCloseTag;
            self.close_tag_buffer.clear();
        } else {
            self.state = TopLevelState::Idle;
        }

        self.json_buffer.clear();
        self.json_phase = JsonParsePhase::SeekingName;
        self.outer_brace_depth = 0;
        self.args_brace_depth = 0;
        self.string_state = StringState::Outside;
        self.name_emitted = false;
        self.current_name.clear();
    }

    /// Process a character while consuming a `</tool_call>` closing tag.
    fn process_close_tag_char(&mut self, ch: char, events: &mut Vec<ToolParserEvent>) {
        self.close_tag_buffer.push(ch);

        // Full close tag consumed
        if self.close_tag_buffer == "</tool_call>" {
            self.close_tag_buffer.clear();
            self.state = TopLevelState::Idle;
            return;
        }

        // Still a valid prefix of the closing tag
        if "</tool_call>".starts_with(&self.close_tag_buffer) {
            return;
        }

        // Not a closing tag -- flush buffer as content and switch to idle
        let text = std::mem::take(&mut self.close_tag_buffer);
        events.push(ToolParserEvent::ContentDelta(text));
        self.state = TopLevelState::Idle;
    }
}

// ---------------------------------------------------------------------------
// JSON string helpers
// ---------------------------------------------------------------------------

/// Find the start of the string value for a given key in a JSON fragment.
///
/// Looks for `"key"` followed by `:` and `"`, returns the position of the
/// first character after the opening quote of the value.
fn find_json_string_value(buffer: &str, key: &str) -> Option<usize> {
    let pattern = format!("\"{}\"", key);
    let key_pos = buffer.find(&pattern)?;

    // After the key, skip whitespace and the `:`, then find opening `"`
    let after_key = key_pos + pattern.len();
    let rest = &buffer[after_key..];

    let mut found_colon = false;
    for (i, ch) in rest.char_indices() {
        if ch == ':' {
            found_colon = true;
        } else if found_colon && ch == '"' {
            return Some(after_key + i + 1); // position after opening "
        } else if found_colon && !ch.is_whitespace() {
            // Non-quote non-whitespace after colon -- value is not a string
            return None;
        }
    }

    None
}

/// Find the start of any JSON value for a given key.
///
/// Similar to `find_json_string_value` but works for any value type
/// (object, array, string, number, etc.).
fn find_json_value_start(buffer: &str, key: &str) -> Option<usize> {
    let pattern = format!("\"{}\"", key);
    let key_pos = buffer.find(&pattern)?;

    let after_key = key_pos + pattern.len();
    let rest = &buffer[after_key..];

    let mut found_colon = false;
    for (i, ch) in rest.char_indices() {
        if ch == ':' {
            found_colon = true;
        } else if found_colon && !ch.is_whitespace() {
            return Some(after_key + i);
        }
    }

    None
}

/// Generate a unique tool call ID.
pub fn generate_tool_call_id() -> String {
    use rand::Rng;
    let mut rng = rand::thread_rng();
    let suffix: String = (0..24)
        .map(|_| format!("{:x}", rng.gen::<u8>() % 16))
        .collect();
    format!("call_{}", suffix)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_inactive_parser_passes_through() {
        let mut parser = ToolCallParser::new(false, None);
        let events = parser.feed("Hello world");
        assert_eq!(events, vec![ToolParserEvent::ContentDelta("Hello world".to_string())]);
        assert!(!parser.has_tool_calls());
    }

    #[test]
    fn test_plain_text_no_tool_calls() {
        let mut parser = ToolCallParser::new(true, None);
        let events = parser.feed("The weather is sunny today.");
        // Each character should be emitted as content
        let content: String = events.iter().filter_map(|e| {
            if let ToolParserEvent::ContentDelta(t) = e { Some(t.as_str()) } else { None }
        }).collect();
        assert_eq!(content, "The weather is sunny today.");
        assert!(!parser.has_tool_calls());
    }

    #[test]
    fn test_delimited_tool_call() {
        let mut parser = ToolCallParser::new(true, None);
        let input = "<tool_call>{\"name\": \"get_weather\", \"arguments\": {\"city\": \"London\"}}</tool_call>";
        let mut all_events = Vec::new();
        for ch in input.chars() {
            all_events.extend(parser.feed(&ch.to_string()));
        }
        all_events.extend(parser.finalize());

        // Should have: ToolCallStart, NameDelta, ArgumentsDelta(s), ToolCallEnd
        let starts: Vec<_> = all_events.iter().filter(|e| matches!(e, ToolParserEvent::ToolCallStart { .. })).collect();
        assert_eq!(starts.len(), 1);

        let names: Vec<String> = all_events.iter().filter_map(|e| {
            if let ToolParserEvent::NameDelta { text, .. } = e { Some(text.clone()) } else { None }
        }).collect();
        assert_eq!(names.join(""), "get_weather");

        let args: String = all_events.iter().filter_map(|e| {
            if let ToolParserEvent::ArgumentsDelta { text, .. } = e { Some(text.as_str()) } else { None }
        }).collect();
        assert_eq!(args, "{\"city\": \"London\"}");

        let ends: Vec<_> = all_events.iter().filter(|e| matches!(e, ToolParserEvent::ToolCallEnd { .. })).collect();
        assert_eq!(ends.len(), 1);
        assert!(parser.has_tool_calls());
    }

    #[test]
    fn test_raw_json_tool_call() {
        let mut parser = ToolCallParser::new(true, None);
        let input = "{\"name\": \"search\", \"arguments\": {\"query\": \"rust\"}}";
        let mut all_events = Vec::new();
        for ch in input.chars() {
            all_events.extend(parser.feed(&ch.to_string()));
        }
        all_events.extend(parser.finalize());

        let names: Vec<String> = all_events.iter().filter_map(|e| {
            if let ToolParserEvent::NameDelta { text, .. } = e { Some(text.clone()) } else { None }
        }).collect();
        assert_eq!(names.join(""), "search");
        assert!(parser.has_tool_calls());
    }

    #[test]
    fn test_escaped_quotes_in_arguments() {
        let mut parser = ToolCallParser::new(true, None);
        let input = r#"{"name": "say", "arguments": {"text": "he said \"hello\""}}"#;
        let mut all_events = Vec::new();
        for ch in input.chars() {
            all_events.extend(parser.feed(&ch.to_string()));
        }
        all_events.extend(parser.finalize());

        let args: String = all_events.iter().filter_map(|e| {
            if let ToolParserEvent::ArgumentsDelta { text, .. } = e { Some(text.as_str()) } else { None }
        }).collect();
        assert!(args.contains(r#"\"hello\""#), "Args should contain escaped quotes: {}", args);
        assert!(parser.has_tool_calls());
    }

    #[test]
    fn test_nested_json_in_arguments() {
        let mut parser = ToolCallParser::new(true, None);
        let input = r#"{"name": "process", "arguments": {"data": {"nested": {"deep": true}}}}"#;
        let mut all_events = Vec::new();
        for ch in input.chars() {
            all_events.extend(parser.feed(&ch.to_string()));
        }
        all_events.extend(parser.finalize());

        let args: String = all_events.iter().filter_map(|e| {
            if let ToolParserEvent::ArgumentsDelta { text, .. } = e { Some(text.as_str()) } else { None }
        }).collect();
        assert!(args.contains("nested"), "Should contain nested object: {}", args);
        assert!(parser.has_tool_calls());
    }

    #[test]
    fn test_parallel_tool_calls() {
        let mut parser = ToolCallParser::new(true, None);
        let input = concat!(
            r#"<tool_call>{"name": "fn1", "arguments": {"a": 1}}</tool_call>"#,
            r#"<tool_call>{"name": "fn2", "arguments": {"b": 2}}</tool_call>"#,
        );
        let mut all_events = Vec::new();
        for ch in input.chars() {
            all_events.extend(parser.feed(&ch.to_string()));
        }
        all_events.extend(parser.finalize());

        let starts: Vec<usize> = all_events.iter().filter_map(|e| {
            if let ToolParserEvent::ToolCallStart { index } = e { Some(*index) } else { None }
        }).collect();
        assert_eq!(starts, vec![0, 1]);

        let ends: Vec<usize> = all_events.iter().filter_map(|e| {
            if let ToolParserEvent::ToolCallEnd { index } = e { Some(*index) } else { None }
        }).collect();
        assert_eq!(ends, vec![0, 1]);

        assert_eq!(parser.tool_call_count, 2);
    }

    #[test]
    fn test_malformed_json_no_crash() {
        let mut parser = ToolCallParser::new(true, None);
        let input = r#"<tool_call>{"name": "broken", "arguments": {"unclosed":#;
        let mut all_events = Vec::new();
        for ch in input.chars() {
            all_events.extend(parser.feed(&ch.to_string()));
        }
        // Finalize should handle the incomplete state gracefully
        all_events.extend(parser.finalize());

        // Should not panic, and should have detected a tool call start
        let starts: Vec<_> = all_events.iter().filter(|e| matches!(e, ToolParserEvent::ToolCallStart { .. })).collect();
        assert!(!starts.is_empty());
    }

    #[test]
    fn test_content_before_tool_call() {
        let mut parser = ToolCallParser::new(true, None);
        let input = "Let me check that. <tool_call>{\"name\": \"search\", \"arguments\": {\"q\": \"test\"}}</tool_call>";
        let mut all_events = Vec::new();
        for ch in input.chars() {
            all_events.extend(parser.feed(&ch.to_string()));
        }
        all_events.extend(parser.finalize());

        let content: String = all_events.iter().filter_map(|e| {
            if let ToolParserEvent::ContentDelta(t) = e { Some(t.as_str()) } else { None }
        }).collect();
        assert!(content.contains("Let me check that. "), "Content: {}", content);

        assert!(parser.has_tool_calls());
    }

    #[test]
    fn test_forced_function_match() {
        let mut parser = ToolCallParser::new(true, Some("get_weather".to_string()));
        let input = r#"{"name": "get_weather", "arguments": {"city": "NYC"}}"#;
        let mut all_events = Vec::new();
        for ch in input.chars() {
            all_events.extend(parser.feed(&ch.to_string()));
        }
        all_events.extend(parser.finalize());

        assert!(parser.has_tool_calls());
    }

    #[test]
    fn test_forced_function_no_match() {
        let mut parser = ToolCallParser::new(true, Some("get_weather".to_string()));
        let input = r#"{"name": "wrong_function", "arguments": {"x": 1}}"#;
        let mut all_events = Vec::new();
        for ch in input.chars() {
            all_events.extend(parser.feed(&ch.to_string()));
        }
        all_events.extend(parser.finalize());

        assert!(!parser.has_tool_calls());
    }

    #[test]
    fn test_tool_call_closing_tag_consumed() {
        let mut parser = ToolCallParser::new(true, None);
        // After tool call ends, `</tool_call>` should be consumed, not emitted as content
        let input = "<tool_call>{\"name\": \"f\", \"arguments\": {}}</tool_call>done";
        let mut all_events = Vec::new();
        for ch in input.chars() {
            all_events.extend(parser.feed(&ch.to_string()));
        }
        all_events.extend(parser.finalize());

        let _content: String = all_events.iter().filter_map(|e| {
            if let ToolParserEvent::ContentDelta(t) = e { Some(t.as_str()) } else { None }
        }).collect();
        // Verify no crash and tool call was detected. The closing </tool_call>
        // tag is consumed by the ConsumingCloseTag state, not emitted as content.
        assert!(parser.has_tool_calls());
    }

    #[test]
    fn test_empty_token() {
        let mut parser = ToolCallParser::new(true, None);
        let events = parser.feed("");
        assert!(events.is_empty());
    }

    #[test]
    fn test_generate_tool_call_id_format() {
        let id = generate_tool_call_id();
        assert!(id.starts_with("call_"));
        assert_eq!(id.len(), 5 + 24); // "call_" + 24 hex chars
    }

    #[test]
    fn test_find_json_string_value() {
        let buf = r#"{"name": "hello"}"#;
        let pos = find_json_string_value(buf, "name");
        assert!(pos.is_some());
        let start = pos.unwrap();
        assert_eq!(&buf[start..start + 5], "hello");
    }

    #[test]
    fn test_find_json_value_start_object() {
        let buf = r#"{"arguments": {"key": "val"}}"#;
        let pos = find_json_value_start(buf, "arguments");
        assert!(pos.is_some());
        let start = pos.unwrap();
        assert_eq!(buf.as_bytes()[start] as char, '{');
    }
}
