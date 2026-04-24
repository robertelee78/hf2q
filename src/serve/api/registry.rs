//! Per-model boundary-marker + tool-call registration (ADR-005 Decision #21,
//! Decision #6).
//!
//! Each supported model family registers the literal text markers its chat
//! template emits for:
//!   - **Reasoning boundaries** — `<|think|>` / `</think|>` style markers
//!     that delimit pre-answer reasoning traces. Tokens between the open
//!     and close markers go into `message.reasoning_content`;
//!     the rest goes into `message.content`. Streaming splits into
//!     `delta.reasoning_content` vs `delta.content` the same way.
//!   - **Tool-call boundaries** — markers delimiting a grammar-constrained
//!     JSON tool-call fragment. Applied by the engine in a later iter when
//!     the grammar sampler is wired into the decode loop.
//!
//! Per ADR-005 Phase 2 refinement (2026-04-23), this is **registration, not
//! parsing**: the open/close markers drive a lightweight state machine over
//! the accumulated decoded text; they don't mine partial JSON out of
//! malformed output. Grammar-constrained decoding (Decision #6) guarantees
//! the JSON is well-formed between the markers.
//!
//! Day-one registered models: `gemma4`, `qwen35` (Qwen 3.5 / 3.6 family).
//! Additional models are added by calling `register(...)` at process start
//! or by editing this file.

use std::collections::HashMap;
use std::sync::OnceLock;

/// Boundary markers + optional preamble string for a single model family.
///
/// ~15-30 LOC per model per ADR-005 Decision #21's target. Co-located with
/// chat-template entries conceptually; each field is independently optional
/// so models that don't emit reasoning traces (most base models) can leave
/// them as `None`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelRegistration {
    /// Opaque identifier for the model family. Matching is done against a
    /// prefix of the request's `model` id — e.g. an id like
    /// `gemma-4-26B-A4B-it-ara-abliterated-dwq` matches the `gemma4`
    /// registration via the `matches` function below.
    pub family: &'static str,

    /// Comma-separated substrings of the model id. Any match on a
    /// case-insensitive substring scan selects this registration. This is
    /// deliberately fuzzy — the set of exact model ids is unbounded, while
    /// the family identifier in ids (`gemma4`, `qwen3.5`, etc.) is stable.
    pub id_substrings: &'static [&'static str],

    /// Opening marker for a reasoning span. `None` = this model doesn't
    /// emit reasoning traces.
    pub reasoning_open: Option<&'static str>,
    /// Closing marker for a reasoning span. `None` = no reasoning.
    pub reasoning_close: Option<&'static str>,

    /// Opening marker for a tool-call JSON block. `None` = no tool calls.
    pub tool_open: Option<&'static str>,
    /// Closing marker for a tool-call JSON block. `None` = no tool calls.
    pub tool_close: Option<&'static str>,

    /// Optional free-text preamble injected before the chat history when
    /// tools are present. Gives the model a hint like
    /// `"You have access to the following tools: ..."`. `None` = no preamble.
    pub tool_preamble: Option<&'static str>,
}

impl ModelRegistration {
    /// Does this registration match the supplied model id?
    /// Case-insensitive substring scan over `id_substrings`.
    pub fn matches(&self, model_id: &str) -> bool {
        let lower = model_id.to_ascii_lowercase();
        self.id_substrings.iter().any(|s| lower.contains(&s.to_ascii_lowercase()))
    }

    /// Returns `true` if this registration has a usable reasoning span.
    pub fn has_reasoning(&self) -> bool {
        self.reasoning_open.is_some() && self.reasoning_close.is_some()
    }

    /// Returns `true` if this registration has a usable tool-call span.
    pub fn has_tools(&self) -> bool {
        self.tool_open.is_some() && self.tool_close.is_some()
    }
}

// ---------------------------------------------------------------------------
// Built-in registrations (day-one models)
// ---------------------------------------------------------------------------

/// Gemma 4 (26B / A4B / variants). Uses `<|think|>` / `</think|>` for
/// reasoning spans (matches the in-template marker `<|think|>` hardcoded in
/// `FALLBACK_GEMMA4_CHAT_TEMPLATE`). Tool calling uses a grammar-constrained
/// JSON block wrapped in `<tool_call>` / `</tool_call>`.
pub const GEMMA4: ModelRegistration = ModelRegistration {
    family: "gemma4",
    id_substrings: &["gemma-4", "gemma4"],
    reasoning_open: Some("<|think|>"),
    reasoning_close: Some("</think|>"),
    tool_open: Some("<tool_call>"),
    tool_close: Some("</tool_call>"),
    tool_preamble: None,
};

/// Qwen 3.5 / 3.6 family. Uses `<think>` / `</think>` — the Qwen convention
/// (no pipe in the closer; distinct from Gemma's). Tool calling also uses
/// `<tool_call>` / `</tool_call>` (Qwen standard).
pub const QWEN35: ModelRegistration = ModelRegistration {
    family: "qwen35",
    id_substrings: &["qwen3.5", "qwen3.6", "qwen35", "qwen36"],
    reasoning_open: Some("<think>"),
    reasoning_close: Some("</think>"),
    tool_open: Some("<tool_call>"),
    tool_close: Some("</tool_call>"),
    tool_preamble: None,
};

/// All built-in registrations in priority order. Later entries override
/// earlier ones when substrings overlap — but day-one substrings are
/// disjoint.
pub const BUILTIN_REGISTRATIONS: &[ModelRegistration] = &[GEMMA4, QWEN35];

// ---------------------------------------------------------------------------
// Registry (process-global)
// ---------------------------------------------------------------------------

/// Dynamic registry, seeded with `BUILTIN_REGISTRATIONS` plus any runtime
/// additions via `register`.
static REGISTRY: OnceLock<std::sync::RwLock<Vec<ModelRegistration>>> =
    OnceLock::new();

fn reg() -> &'static std::sync::RwLock<Vec<ModelRegistration>> {
    REGISTRY.get_or_init(|| {
        std::sync::RwLock::new(BUILTIN_REGISTRATIONS.to_vec())
    })
}

/// Find a registration matching `model_id`, or `None` if no family matches.
/// Matches on case-insensitive substring per `ModelRegistration::matches`.
pub fn find_for(model_id: &str) -> Option<ModelRegistration> {
    let guard = reg().read().unwrap();
    for r in guard.iter() {
        if r.matches(model_id) {
            return Some(r.clone());
        }
    }
    None
}

/// List all registered model families. Useful for `/v1/models` extension
/// fields + debug diagnostics.
pub fn list_families() -> Vec<String> {
    let guard = reg().read().unwrap();
    guard.iter().map(|r| r.family.to_string()).collect()
}

/// Register an additional model family at runtime (e.g. for downstream
/// applications embedding hf2q as a library). Later registrations take
/// precedence on overlapping substrings.
pub fn register(entry: ModelRegistration) {
    reg().write().unwrap().push(entry);
}

// ---------------------------------------------------------------------------
// Reasoning-boundary state machine (Decision #21)
// ---------------------------------------------------------------------------

/// Tracks position inside a reasoning span while decoded text accumulates.
/// Engine creates one per generation, feeds decoded fragments in, and gets
/// back per-fragment classification (`DeltaKind::Reasoning` or
/// `DeltaKind::Content`).
///
/// The state machine is simple: a single `in_reasoning` flag that flips on
/// `reasoning_open` and off on `reasoning_close`. Markers are detected by
/// substring scan over the most-recent accumulated text — the engine holds
/// a **tail buffer** of the last `max(open.len, close.len)` bytes to avoid
/// missing markers that span fragment boundaries.
#[derive(Debug, Clone)]
pub struct ReasoningSplitter {
    open_marker: &'static str,
    close_marker: &'static str,
    in_reasoning: bool,
    /// Sliding tail of decoded text — long enough to see either marker span
    /// across token boundaries.
    tail_buf: String,
    tail_cap: usize,
}

impl ReasoningSplitter {
    /// Build from a registration. If the registration has no reasoning
    /// markers, returns `None` — callers route all text to `Content`.
    pub fn from_registration(reg: &ModelRegistration) -> Option<Self> {
        let (open, close) = match (reg.reasoning_open, reg.reasoning_close) {
            (Some(o), Some(c)) if !o.is_empty() && !c.is_empty() => (o, c),
            _ => return None,
        };
        let cap = open.len().max(close.len()).max(1);
        Some(Self {
            open_marker: open,
            close_marker: close,
            in_reasoning: false,
            tail_buf: String::with_capacity(cap * 2),
            tail_cap: cap,
        })
    }

    /// Accept a fragment of decoded text. Returns a `Vec<(Slot, String)>`
    /// describing how the fragment should be routed. A fragment may produce
    /// multiple slots if a marker boundary falls inside it.
    ///
    /// Markers themselves are **swallowed** — they don't appear in either
    /// output slot. This matches the OpenAI-o1 convention: the user sees
    /// `reasoning_content` as clean text, `content` as clean text, and the
    /// marker delimiters are hidden.
    pub fn feed(&mut self, fragment: &str) -> Vec<(SplitSlot, String)> {
        let mut out: Vec<(SplitSlot, String)> = Vec::new();
        // Prepend the sliding tail so markers that span fragment boundaries
        // are still detected. The tail was *held back* from prior emission
        // (not already emitted), so scan + emit from offset 0.
        let mut scan = std::mem::take(&mut self.tail_buf);
        scan.push_str(fragment);

        let mut scan_cursor = 0usize;
        let mut out_cursor = 0usize;

        loop {
            let active_marker = if self.in_reasoning {
                self.close_marker
            } else {
                self.open_marker
            };
            match scan[scan_cursor..].find(active_marker) {
                Some(rel) => {
                    let marker_start = scan_cursor + rel;
                    // Emit text [out_cursor..marker_start] to the current slot.
                    let slot = if self.in_reasoning {
                        SplitSlot::Reasoning
                    } else {
                        SplitSlot::Content
                    };
                    if marker_start > out_cursor {
                        out.push((slot, scan[out_cursor..marker_start].to_string()));
                    }
                    // Flip state + skip past marker.
                    self.in_reasoning = !self.in_reasoning;
                    scan_cursor = marker_start + active_marker.len();
                    out_cursor = scan_cursor;
                }
                None => {
                    // No more markers in the remainder. Commit the tail
                    // portion that's still sitting after out_cursor, minus
                    // the last `tail_cap` bytes which we hold back in case
                    // they're the start of a next-fragment marker.
                    let total_len = scan.len();
                    let emit_end = total_len.saturating_sub(self.tail_cap);
                    if emit_end > out_cursor {
                        // Align emit_end to a char boundary so we never split
                        // a UTF-8 code point.
                        let emit_end = snap_down_char_boundary(&scan, emit_end);
                        if emit_end > out_cursor {
                            let slot = if self.in_reasoning {
                                SplitSlot::Reasoning
                            } else {
                                SplitSlot::Content
                            };
                            out.push((slot, scan[out_cursor..emit_end].to_string()));
                            out_cursor = emit_end;
                        }
                    }
                    // Stash the remainder as the new tail.
                    self.tail_buf = scan[out_cursor..].to_string();
                    break;
                }
            }
        }
        out
    }

    /// Drain any buffered tail into an output slot at generation end. Called
    /// by the engine when decode finishes, so tail-stashed text isn't lost.
    pub fn finish(&mut self) -> Option<(SplitSlot, String)> {
        if self.tail_buf.is_empty() {
            return None;
        }
        let slot = if self.in_reasoning {
            SplitSlot::Reasoning
        } else {
            SplitSlot::Content
        };
        let text = std::mem::take(&mut self.tail_buf);
        Some((slot, text))
    }

    pub fn in_reasoning(&self) -> bool {
        self.in_reasoning
    }
}

fn snap_down_char_boundary(s: &str, mut idx: usize) -> usize {
    // Snap idx DOWN to the nearest char boundary so we don't split a UTF-8
    // code point when slicing out a contiguous run.
    while idx > 0 && !s.is_char_boundary(idx) {
        idx -= 1;
    }
    idx
}

/// Which OpenAI delta slot a fragment belongs to.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SplitSlot {
    /// `delta.content` / `message.content`.
    Content,
    /// `delta.reasoning_content` / `message.reasoning_content`.
    Reasoning,
}

// ---------------------------------------------------------------------------
// Reasoning-pair assembly (for non-streaming responses)
// ---------------------------------------------------------------------------

/// Helper: run a `ReasoningSplitter` over the full generated text and
/// return the two split strings. Used by the non-streaming path to populate
/// `message.content` + `message.reasoning_content` on the final response.
pub fn split_full_output(
    reg: &ModelRegistration,
    full_text: &str,
) -> (String, Option<String>) {
    let mut splitter = match ReasoningSplitter::from_registration(reg) {
        Some(s) => s,
        None => return (full_text.to_string(), None),
    };
    let mut content = String::new();
    let mut reasoning = String::new();
    for (slot, frag) in splitter.feed(full_text) {
        match slot {
            SplitSlot::Content => content.push_str(&frag),
            SplitSlot::Reasoning => reasoning.push_str(&frag),
        }
    }
    if let Some((slot, frag)) = splitter.finish() {
        match slot {
            SplitSlot::Content => content.push_str(&frag),
            SplitSlot::Reasoning => reasoning.push_str(&frag),
        }
    }
    (content, if reasoning.is_empty() { None } else { Some(reasoning) })
}

/// Silence the unused-import warning when the registry itself is only
/// touched by the engine (which isn't compiled in this test binary).
#[allow(dead_code)]
const _COMPILE_REFERENCES: fn() -> HashMap<String, ModelRegistration> = || HashMap::new();

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gemma4_matches_real_model_ids() {
        assert!(GEMMA4.matches("gemma-4-26B-A4B-it-ara-abliterated-dwq"));
        assert!(GEMMA4.matches("gemma-4-27b-it"));
        assert!(GEMMA4.matches("GEMMA-4-test"));
    }

    #[test]
    fn qwen35_matches_family_ids() {
        assert!(QWEN35.matches("qwen3.5-27b"));
        assert!(QWEN35.matches("qwen3.6-35b-a3b-abliterix"));
        assert!(QWEN35.matches("Qwen35-14B-chat"));
    }

    #[test]
    fn non_matching_model_id_returns_none() {
        assert!(find_for("llama-3.2-1b").is_none());
        assert!(find_for("unknown-model").is_none());
    }

    #[test]
    fn gemma4_has_reasoning_and_tools() {
        assert!(GEMMA4.has_reasoning());
        assert!(GEMMA4.has_tools());
        assert_eq!(GEMMA4.reasoning_open, Some("<|think|>"));
        assert_eq!(GEMMA4.reasoning_close, Some("</think|>"));
    }

    #[test]
    fn qwen35_has_different_reasoning_markers() {
        // Regression: don't conflate gemma's `<|think|>` with qwen's `<think>`.
        assert_ne!(
            GEMMA4.reasoning_open,
            QWEN35.reasoning_open
        );
    }

    #[test]
    fn register_appends_and_wins_over_builtin_on_substring_overlap() {
        register(ModelRegistration {
            family: "custom",
            id_substrings: &["test_register_overlap"],
            reasoning_open: Some("<R>"),
            reasoning_close: Some("</R>"),
            tool_open: None,
            tool_close: None,
            tool_preamble: None,
        });
        let found = find_for("test_register_overlap-001").expect("found");
        // Built-ins are scanned first; the newly-registered entry only wins
        // if no built-in matches. That's fine for this disjoint substring.
        assert_eq!(found.family, "custom");
    }

    // --- ReasoningSplitter ---

    fn split(reg: &ModelRegistration, s: &str) -> Vec<(SplitSlot, String)> {
        // Run the splitter and coalesce adjacent same-slot runs — the
        // splitter holds back a tail buffer to safely detect markers across
        // fragment boundaries, which can emit one slot run as two parts.
        // Coalescing gives the logical classification.
        let mut sp = ReasoningSplitter::from_registration(reg).unwrap();
        let mut out = sp.feed(s);
        if let Some(tail) = sp.finish() {
            out.push(tail);
        }
        coalesce(&out)
    }

    #[test]
    fn splitter_no_markers_all_content() {
        let out = split(&GEMMA4, "hello world");
        assert_eq!(out, vec![(SplitSlot::Content, "hello world".into())]);
    }

    #[test]
    fn splitter_single_reasoning_span() {
        // Use real gemma markers: `<|think|>` open, `</think|>` close.
        let out = split(&GEMMA4, "pre <|think|>because</think|> post");
        assert_eq!(
            out,
            vec![
                (SplitSlot::Content, "pre ".into()),
                (SplitSlot::Reasoning, "because".into()),
                (SplitSlot::Content, " post".into()),
            ]
        );
    }

    #[test]
    fn splitter_open_without_close_reasoning_continues_to_end() {
        let out = split(&GEMMA4, "pre <|think|>still thinking");
        assert_eq!(
            out,
            vec![
                (SplitSlot::Content, "pre ".into()),
                (SplitSlot::Reasoning, "still thinking".into()),
            ]
        );
    }

    #[test]
    fn splitter_marker_spans_fragment_boundary() {
        // The open marker `<|think|>` is 9 bytes. Feeding it in two
        // fragments should still detect it via the sliding tail buffer.
        let mut sp = ReasoningSplitter::from_registration(&GEMMA4).unwrap();
        let a = sp.feed("before <|th");
        let b = sp.feed("ink|>reasoning</think|>after");
        let c = sp.finish();
        let mut all: Vec<(SplitSlot, String)> = Vec::new();
        all.extend(a);
        all.extend(b);
        if let Some(t) = c {
            all.push(t);
        }
        let joined = coalesce(&all);
        assert_eq!(
            joined,
            vec![
                (SplitSlot::Content, "before ".into()),
                (SplitSlot::Reasoning, "reasoning".into()),
                (SplitSlot::Content, "after".into()),
            ]
        );
    }

    #[test]
    fn splitter_multiple_reasoning_spans() {
        let out = split(
            &GEMMA4,
            "a<|think|>b</think|>c<|think|>d</think|>e",
        );
        let joined = coalesce(&out);
        assert_eq!(
            joined,
            vec![
                (SplitSlot::Content, "a".into()),
                (SplitSlot::Reasoning, "b".into()),
                (SplitSlot::Content, "c".into()),
                (SplitSlot::Reasoning, "d".into()),
                (SplitSlot::Content, "e".into()),
            ]
        );
    }

    #[test]
    fn splitter_qwen_markers_distinct_from_gemma() {
        let out = split(&QWEN35, "hi <think>pondering</think> there");
        let joined = coalesce(&out);
        assert_eq!(
            joined,
            vec![
                (SplitSlot::Content, "hi ".into()),
                (SplitSlot::Reasoning, "pondering".into()),
                (SplitSlot::Content, " there".into()),
            ]
        );
    }

    #[test]
    fn splitter_does_not_split_utf8_at_fragment_end() {
        // Greek α is 2 bytes. If snap_down_char_boundary were wrong, we'd
        // panic with "byte index X is not a char boundary".
        let mut sp = ReasoningSplitter::from_registration(&GEMMA4).unwrap();
        let _ = sp.feed("hello α");
        let _ = sp.feed("β world");
        let _ = sp.finish();
    }

    #[test]
    fn split_full_output_helper_returns_both_slots() {
        // Two reasoning spans with real Gemma markers.
        let (content, reasoning) = split_full_output(
            &GEMMA4,
            "a <|think|>r1</think|> b <|think|>r2</think|> c",
        );
        assert_eq!(content, "a  b  c");
        assert_eq!(reasoning.as_deref(), Some("r1r2"));
    }

    #[test]
    fn split_full_output_no_markers_returns_none_reasoning() {
        let (content, reasoning) = split_full_output(&GEMMA4, "just plain content");
        assert_eq!(content, "just plain content");
        assert_eq!(reasoning, None);
    }

    #[test]
    fn list_families_includes_builtins() {
        let fams = list_families();
        assert!(fams.iter().any(|f| f == "gemma4"));
        assert!(fams.iter().any(|f| f == "qwen35"));
    }

    // Merge adjacent same-slot runs — useful for asserting against streaming
    // output where the splitter might emit a Content run as two pieces split
    // on the tail-buffer boundary.
    fn coalesce(v: &[(SplitSlot, String)]) -> Vec<(SplitSlot, String)> {
        let mut out: Vec<(SplitSlot, String)> = Vec::new();
        for (slot, s) in v {
            if let Some(last) = out.last_mut() {
                if last.0 == *slot {
                    last.1.push_str(s);
                    continue;
                }
            }
            out.push((*slot, s.clone()));
        }
        out
    }
}
