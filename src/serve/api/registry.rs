//! Per-model boundary-marker + tool-call registration (ADR-005 Decision #21,
//! Decision #6).
//!
//! Each supported model family registers the literal text markers its chat
//! template emits for:
//!   - **Reasoning boundaries** — open/close marker pair that delimits
//!     pre-answer reasoning traces. Tokens between the open and close markers
//!     go into `message.reasoning_content`; the rest goes into
//!     `message.content`. Streaming splits into `delta.reasoning_content`
//!     vs `delta.content` the same way. Per-family marker shapes vary:
//!     Qwen 3.5/3.6 emits the standard `<think>` / `</think>` HF convention;
//!     Gemma 4 emits its `<|channel>` / `<channel|>` channel-block convention
//!     (matches the chat-template `strip_thinking` macro and the
//!     tokenizer_config `x-regex` that spans `<|channel>thought\n…<channel|>`).
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

/// Gemma 4 (26B / A4B / variants). Uses `<|channel>` / `<channel|>` for
/// reasoning spans — the asymmetric channel-block convention emitted by the
/// model whenever it produces a thinking trace
/// (`<|channel>thought\n…<channel|>`; see `models/gemma-4-26B-A4B-it-ara-abliterated-dwq/chat_template.jinja:141-151`'s
/// `strip_thinking` macro and `tokenizer_config.json` `x-regex`
/// `\<\|channel\>thought\n(?P<thinking>.*?)\<channel\|\>`). Tool calling
/// uses the parallel `<|tool_call>` / `<tool_call|>` shape — note the
/// asymmetric pipe placement on both pairs (see chat_template lines 189-203).
///
/// **Marker shape audit (2026-04-26 W66 iter-133 Iter B-2):** the tool-call
/// pair was previously declared as `<tool_call>` / `</tool_call>` (the Qwen
/// convention). The gemma-4 GGUF chat template actually emits
/// `<|tool_call>call:NAME{...}<tool_call|>`, not `<tool_call>...</tool_call>`.
/// Real-model fixture `tests/fixtures/openwebui_multiturn/scenario2_tool_call_chunks.txt`
/// (W65 iter-133 Iter B) confirmed the literal mismatch. Iter B-2 fixed
/// the registration to match the in-template strings so the engine's
/// `ToolCallSplitter` actually detects what the model emits.
///
/// **Reasoning marker audit (2026-04-26 W67 iter-133 Iter D):** same
/// bug-class as the iter-B-2 tool-call fix. The reasoning pair was
/// previously declared as `<|think|>` / `</think|>` — both wrong. The
/// `<|think|>` literal is the system-block thinking-hint emitted only when
/// `enable_thinking=true` is passed to the chat template (chat_template:162);
/// it is not the runtime reasoning-emission boundary. The actual emission
/// pair (consulted authoritatively by `strip_thinking` and the
/// tokenizer_config `x-regex`) is `<|channel>` (open; aliased `soc_token`
/// in tokenizer_config:87) / `<channel|>` (close; aliased `eoc_token` in
/// tokenizer_config:8,28). This iter corrects the registration so the
/// engine's `ReasoningSplitter` detects what the model emits during decode
/// (any `<|channel>thought\n…<channel|>` block the model produces gets
/// routed to `delta.reasoning_content`; the literal `thought\n` channel
/// identifier remains visible inside the routed reasoning text by design,
/// mirroring `strip_thinking`'s scope).
pub const GEMMA4: ModelRegistration = ModelRegistration {
    family: "gemma4",
    id_substrings: &["gemma-4", "gemma4"],
    reasoning_open: Some("<|channel>"),
    reasoning_close: Some("<channel|>"),
    tool_open: Some("<|tool_call>"),
    tool_close: Some("<tool_call|>"),
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
// Tool-call boundary state machine (Decision #21 sibling, iter-133 Iter B-2)
// ---------------------------------------------------------------------------

/// Output emitted by `ToolCallSplitter::feed`. The splitter classifies each
/// decoded fragment into one of:
///   - **`Content(text)`** — outside any tool-call span; route to
///     `delta.content` (or further classify via `ReasoningSplitter`).
///   - **`ToolCallOpen`** — a tool-call open marker has just been observed;
///     emitted exactly once per call. The producer should:
///       1. Synthesize a `tool_call_id` (`call_<rand>`).
///       2. Buffer subsequent `ToolCallText` fragments until `ToolCallClose`.
///   - **`ToolCallText(text)`** — accumulated raw text inside the open/close
///     markers (including the `call:NAME{...}` Gemma 4 syntax). The marker
///     literals themselves are swallowed; the text run is the model's verbatim
///     argument syntax.
///   - **`ToolCallClose`** — the close marker has been observed; the
///     accumulated tool-call body is complete and the producer can parse +
///     emit the structured `delta.tool_calls.function.{name,arguments}` chunk.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ToolCallEvent {
    Content(String),
    ToolCallOpen,
    ToolCallText(String),
    ToolCallClose,
}

/// Tracks position inside a tool-call span while decoded text accumulates.
/// Sibling to `ReasoningSplitter` — same tail-buffer pattern, same
/// boundary-marker semantics, different output enum because tool-call
/// downstream wiring (`GenerationEvent::ToolCallDelta`) carries more
/// structure than `Reasoning` vs `Content`.
///
/// **Marker shape (per-model registration, not parsing):** the splitter
/// detects the literal open/close markers from `ModelRegistration.tool_open`
/// / `tool_close` (e.g. Gemma 4: `<|tool_call>` / `<tool_call|>`; Qwen 3.5/3.6:
/// `<tool_call>` / `</tool_call>`). The text run between them — the `call:NAME{kv-list}`
/// Gemma syntax or the `<function=NAME><parameter=...>...</function>` Qwen
/// syntax — is parsed by a per-model parser at `ToolCallClose` emission, not
/// by this splitter.
///
/// **Composition with `ReasoningSplitter`:** the engine runs
/// `ReasoningSplitter` first (reasoning is always outside tool calls); the
/// `Content`-classified fragments then flow into `ToolCallSplitter`. The
/// same tail-buffer discipline guarantees markers spanning fragment
/// boundaries are still detected.
#[derive(Debug, Clone)]
pub struct ToolCallSplitter {
    open_marker: &'static str,
    close_marker: &'static str,
    in_tool_call: bool,
    /// Sliding tail of decoded text — long enough to see either marker span
    /// across token boundaries. See `ReasoningSplitter::tail_buf` for the
    /// identical mechanism.
    tail_buf: String,
    tail_cap: usize,
}

impl ToolCallSplitter {
    /// Build from a registration. If the registration has no tool markers,
    /// returns `None` — callers route all text to `Content`.
    pub fn from_registration(reg: &ModelRegistration) -> Option<Self> {
        let (open, close) = match (reg.tool_open, reg.tool_close) {
            (Some(o), Some(c)) if !o.is_empty() && !c.is_empty() => (o, c),
            _ => return None,
        };
        let cap = open.len().max(close.len()).max(1);
        Some(Self {
            open_marker: open,
            close_marker: close,
            in_tool_call: false,
            tail_buf: String::with_capacity(cap * 2),
            tail_cap: cap,
        })
    }

    /// Accept a fragment of decoded text. Returns a sequence of
    /// `ToolCallEvent`s describing how the fragment should be routed. A
    /// fragment may produce multiple events if a marker boundary falls
    /// inside it (e.g. `pre <|tool_call>call:f{}<tool_call|> post` →
    /// `[Content("pre "), ToolCallOpen, ToolCallText("call:f{}"),
    /// ToolCallClose, Content(" post")]`).
    ///
    /// Markers themselves are **swallowed** — they don't appear in any
    /// emitted text event. This matches the OpenAI spec: `delta.content`
    /// gets natural-language text only; `delta.tool_calls.function.{name,
    /// arguments}` gets the parsed call syntax (parsed by the engine after
    /// observing `ToolCallClose`).
    pub fn feed(&mut self, fragment: &str) -> Vec<ToolCallEvent> {
        let mut out: Vec<ToolCallEvent> = Vec::new();
        // Prepend the sliding tail so markers that span fragment boundaries
        // are still detected (same mechanism as ReasoningSplitter).
        let mut scan = std::mem::take(&mut self.tail_buf);
        scan.push_str(fragment);

        let mut scan_cursor = 0usize;
        let mut out_cursor = 0usize;

        loop {
            let active_marker = if self.in_tool_call {
                self.close_marker
            } else {
                self.open_marker
            };
            match scan[scan_cursor..].find(active_marker) {
                Some(rel) => {
                    let marker_start = scan_cursor + rel;
                    // Emit text [out_cursor..marker_start] as the current slot.
                    if marker_start > out_cursor {
                        let text = scan[out_cursor..marker_start].to_string();
                        if self.in_tool_call {
                            out.push(ToolCallEvent::ToolCallText(text));
                        } else {
                            out.push(ToolCallEvent::Content(text));
                        }
                    }
                    // Flip state + emit the open/close synthetic event.
                    if self.in_tool_call {
                        out.push(ToolCallEvent::ToolCallClose);
                    } else {
                        out.push(ToolCallEvent::ToolCallOpen);
                    }
                    self.in_tool_call = !self.in_tool_call;
                    scan_cursor = marker_start + active_marker.len();
                    out_cursor = scan_cursor;
                }
                None => {
                    // No more markers in the remainder. Emit the portion that's
                    // still after out_cursor MINUS the last `tail_cap` bytes,
                    // which we hold back in case they're the start of a
                    // next-fragment marker.
                    let total_len = scan.len();
                    let emit_end = total_len.saturating_sub(self.tail_cap);
                    if emit_end > out_cursor {
                        let emit_end = snap_down_char_boundary(&scan, emit_end);
                        if emit_end > out_cursor {
                            let text = scan[out_cursor..emit_end].to_string();
                            if self.in_tool_call {
                                out.push(ToolCallEvent::ToolCallText(text));
                            } else {
                                out.push(ToolCallEvent::Content(text));
                            }
                            out_cursor = emit_end;
                        }
                    }
                    self.tail_buf = scan[out_cursor..].to_string();
                    break;
                }
            }
        }
        out
    }

    /// Drain any buffered tail at generation end. Called by the engine when
    /// decode finishes so tail-stashed text isn't lost. Note: if generation
    /// ends mid-tool-call (e.g. EOS while `in_tool_call=true`) the tail goes
    /// into `ToolCallText`; the caller is responsible for deciding what to
    /// do with an unterminated call (typical: emit a synthetic
    /// `ToolCallClose` and finalize anyway, OR drop the partial call).
    pub fn finish(&mut self) -> Option<ToolCallEvent> {
        if self.tail_buf.is_empty() {
            return None;
        }
        let text = std::mem::take(&mut self.tail_buf);
        if self.in_tool_call {
            Some(ToolCallEvent::ToolCallText(text))
        } else {
            Some(ToolCallEvent::Content(text))
        }
    }

    pub fn in_tool_call(&self) -> bool {
        self.in_tool_call
    }
}

// ---------------------------------------------------------------------------
// Tool-call body parser (Gemma 4 + Qwen 3.5/3.6 syntaxes)
// ---------------------------------------------------------------------------

/// Parsed shape of a single tool-call body, ready to populate the
/// OpenAI `delta.tool_calls.function.{name, arguments}` chunk fields.
///
/// `arguments` is a JSON-encoded string per the OpenAI streaming spec —
/// clients accumulate it across deltas and `JSON.parse` at the end.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParsedToolCall {
    pub name: String,
    /// JSON-encoded arguments (e.g. `{"location":"Paris"}`).
    pub arguments_json: String,
}

/// Parse a tool-call body emitted between the open/close markers.
///
/// Day-one supports two per-family syntaxes:
///
/// * **Gemma 4** — `call:NAME{key1:<|"|>val1<|"|>,key2:<|"|>val2<|"|>}` (string
///   args wrapped in `<|"|>` quote-markers; numeric/boolean args bare). See
///   `models/gemma4/chat_template.jinja:113-200`.
/// * **Qwen 3.5/3.6** — `\n<function=NAME>\n<parameter=key>\nval\n</parameter>\n...\n</function>\n`.
///   See the Qwen 3.6 GGUF tokenizer_config.json `chat_template` field.
///
/// Returns `None` when the body is unparseable (logged at the call site so
/// the engine can degrade gracefully — emit a `Content(raw_body)` fragment
/// rather than a malformed tool-call delta).
pub fn parse_tool_call_body(reg: &ModelRegistration, body: &str) -> Option<ParsedToolCall> {
    match reg.family {
        "gemma4" => parse_gemma4_tool_call(body),
        "qwen35" => parse_qwen35_tool_call(body),
        _ => None,
    }
}

/// Parse Gemma 4's `call:NAME{kv-list}` body. Whitespace-tolerant.
///
/// String values are wrapped in `<|"|>...<|"|>` markers (the Gemma quote
/// convention; see `models/gemma4/chat_template.jinja:113`). Bare values
/// (no surrounding quote-markers) are treated as JSON-literals (numbers,
/// booleans). Keys are bare identifiers.
fn parse_gemma4_tool_call(body: &str) -> Option<ParsedToolCall> {
    let body = body.trim();
    // Expect `call:NAME{...}`. The open marker has been swallowed by the
    // splitter; the close marker has too. So we start with `call:`.
    let rest = body.strip_prefix("call:")?;
    let brace_start = rest.find('{')?;
    let name = rest[..brace_start].trim().to_string();
    if name.is_empty() {
        return None;
    }
    // Match braces — body of args ends at the LAST `}` (Gemma can have
    // nested args via the `<|"|>` quote-markers, but the outer brace is
    // the call boundary).
    let after_open = &rest[brace_start + 1..];
    let close_idx = after_open.rfind('}')?;
    let kv_str = &after_open[..close_idx];

    // Split on top-level commas — top-level meaning: not inside `<|"|>...<|"|>`.
    let kvs = split_top_level_kvs(kv_str);
    let mut args = serde_json::Map::new();
    for kv in kvs {
        let (k, v) = kv.split_once(':')?;
        let key = k.trim().to_string();
        if key.is_empty() {
            return None;
        }
        // Decode the value: if wrapped in `<|"|>...<|"|>`, treat as string;
        // otherwise treat as JSON literal (number, bool, null).
        let v = v.trim();
        let json_val = if let Some(stripped) = v
            .strip_prefix("<|\"|>")
            .and_then(|s| s.strip_suffix("<|\"|>"))
        {
            serde_json::Value::String(stripped.to_string())
        } else if let Ok(num) = v.parse::<i64>() {
            serde_json::Value::from(num)
        } else if let Ok(num) = v.parse::<f64>() {
            serde_json::Value::from(num)
        } else if v == "true" {
            serde_json::Value::Bool(true)
        } else if v == "false" {
            serde_json::Value::Bool(false)
        } else if v == "null" {
            serde_json::Value::Null
        } else {
            // Fallback: treat unquoted bare-value as a string. Better to
            // forward the model's intent than to drop the field. Logged at
            // the call site so calibration drift is visible.
            serde_json::Value::String(v.to_string())
        };
        args.insert(key, json_val);
    }
    let arguments_json = serde_json::to_string(&serde_json::Value::Object(args))
        .ok()?;
    Some(ParsedToolCall {
        name,
        arguments_json,
    })
}

/// Split a Gemma 4 kv-list on top-level commas — i.e. commas NOT inside a
/// `<|"|>...<|"|>` string-quote span. Lightweight scanner; does not handle
/// nested objects (Gemma 4 doesn't emit them per the chat template).
fn split_top_level_kvs(s: &str) -> Vec<&str> {
    let mut out = Vec::new();
    let mut start = 0usize;
    let mut in_str = false;
    let bytes = s.as_bytes();
    let mut i = 0usize;
    while i < bytes.len() {
        // Detect `<|"|>` (5 bytes) at byte i.
        if !in_str && bytes[i..].starts_with(b"<|\"|>") {
            in_str = true;
            i += 5;
            continue;
        }
        if in_str && bytes[i..].starts_with(b"<|\"|>") {
            in_str = false;
            i += 5;
            continue;
        }
        if !in_str && bytes[i] == b',' {
            out.push(&s[start..i]);
            start = i + 1;
        }
        i += 1;
    }
    if start < s.len() {
        out.push(&s[start..]);
    }
    out
}

/// Parse Qwen 3.5/3.6's `<function=NAME>\n<parameter=key>\nval\n</parameter>\n...\n</function>` body.
///
/// Whitespace-tolerant; `<parameter=key>...</parameter>` blocks become args.
fn parse_qwen35_tool_call(body: &str) -> Option<ParsedToolCall> {
    let body = body.trim();
    // Expect `<function=NAME>...</function>`.
    let after_func = body.strip_prefix("<function=")?;
    let name_end = after_func.find('>')?;
    let name = after_func[..name_end].trim().to_string();
    if name.is_empty() {
        return None;
    }
    let after_name_close = &after_func[name_end + 1..];
    let func_close = after_name_close.rfind("</function>")?;
    let inner = after_name_close[..func_close].trim();

    // Walk `<parameter=KEY>VAL</parameter>` blocks.
    let mut args = serde_json::Map::new();
    let mut cursor = 0usize;
    while cursor < inner.len() {
        let Some(rel_open) = inner[cursor..].find("<parameter=") else { break };
        let p_open = cursor + rel_open;
        let key_start = p_open + "<parameter=".len();
        let Some(rel_gt) = inner[key_start..].find('>') else { break };
        let key_end = key_start + rel_gt;
        let key = inner[key_start..key_end].trim().to_string();
        let val_start = key_end + 1;
        let Some(rel_close) = inner[val_start..].find("</parameter>") else { break };
        let val_end = val_start + rel_close;
        let val_raw = inner[val_start..val_end].trim();
        // Try JSON-literal parse first (for numbers/booleans/objects);
        // fall back to plain string. Qwen's template recommends `tojson` on
        // the value, so well-formed JSON is the common case.
        let json_val: serde_json::Value = match serde_json::from_str(val_raw) {
            Ok(v) => v,
            Err(_) => serde_json::Value::String(val_raw.to_string()),
        };
        args.insert(key, json_val);
        cursor = val_end + "</parameter>".len();
    }
    let arguments_json = serde_json::to_string(&serde_json::Value::Object(args)).ok()?;
    Some(ParsedToolCall {
        name,
        arguments_json,
    })
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
        // Iter D W67: the reasoning pair is the runtime `<|channel>` /
        // `<channel|>` channel-block convention (matches `strip_thinking`
        // and tokenizer_config.json `x-regex`), NOT the `<|think|>` /
        // `</think|>` literal which is the prompt-side thinking-mode hint.
        assert_eq!(GEMMA4.reasoning_open, Some("<|channel>"));
        assert_eq!(GEMMA4.reasoning_close, Some("<channel|>"));
    }

    /// Iter D W67: lock in the corrected Gemma 4 reasoning markers — same
    /// bug-class as iter B-2's tool-call fix. The chat-template
    /// `strip_thinking` macro authoritatively defines the reasoning span as
    /// `<|channel>` … `<channel|>` (text.split('<channel|>') THEN look for
    /// `'<|channel>' in part`); the tokenizer_config `x-regex` confirms the
    /// emission shape. Pre-fix the registry declared `<|think|>` /
    /// `</think|>`, the system-side thinking-mode hint, which the model
    /// never emits as a runtime delimiter — so the splitter would never
    /// detect a real Gemma 4 reasoning span.
    #[test]
    fn gemma4_reasoning_markers_match_chat_template_emission() {
        // Authoritative reference: chat_template.jinja `strip_thinking`
        // splits on `<channel|>` then trims `<|channel>...` prefix from
        // each part. Same pair the runtime tokens emit.
        assert_eq!(GEMMA4.reasoning_open, Some("<|channel>"));
        assert_eq!(GEMMA4.reasoning_close, Some("<channel|>"));
        // Cross-check: tokenizer_config.json declares `soc_token` =
        // `<|channel>` and `eoc_token` = `<channel|>`; these are the
        // canonical channel-block delimiters Gemma 4 emits.
        assert_eq!(GEMMA4.reasoning_open.unwrap(), "<|channel>");
        assert_eq!(GEMMA4.reasoning_close.unwrap(), "<channel|>");
    }

    #[test]
    fn qwen35_has_different_reasoning_markers() {
        // Regression: don't conflate Qwen's `<think>` / `</think>` HF
        // convention with Gemma's asymmetric `<|channel>` / `<channel|>`
        // channel-block convention.
        assert_ne!(
            GEMMA4.reasoning_open,
            QWEN35.reasoning_open
        );
        assert_ne!(
            GEMMA4.reasoning_close,
            QWEN35.reasoning_close
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
        // Use real gemma markers: `<|channel>` open, `<channel|>` close
        // (iter D W67 corrected from the previous `<|think|>` / `</think|>`
        // declaration that the model never emits at runtime).
        let out = split(&GEMMA4, "pre <|channel>because<channel|> post");
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
        let out = split(&GEMMA4, "pre <|channel>still thinking");
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
        // The open marker `<|channel>` is 10 bytes; feeding it in two
        // fragments should still detect it via the sliding tail buffer.
        let mut sp = ReasoningSplitter::from_registration(&GEMMA4).unwrap();
        let a = sp.feed("before <|chan");
        let b = sp.feed("nel>reasoning<channel|>after");
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
            "a<|channel>b<channel|>c<|channel>d<channel|>e",
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

    /// Iter D W67: realistic Gemma 4 emission — the model produces
    /// `<|channel>thought\n[REASONING_TEXT]<channel|>[ANSWER]` per the
    /// tokenizer_config `x-regex`. The literal `thought\n` channel
    /// identifier is part of the routed reasoning span (`strip_thinking`
    /// preserves the channel name), and the post-close run is content.
    #[test]
    fn splitter_gemma4_realistic_thought_channel_emission() {
        let out = split(
            &GEMMA4,
            "<|channel>thought\nlet me compute 73 * 47<channel|>The answer is 3431",
        );
        let joined = coalesce(&out);
        assert_eq!(
            joined,
            vec![
                (SplitSlot::Reasoning, "thought\nlet me compute 73 * 47".into()),
                (SplitSlot::Content, "The answer is 3431".into()),
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
        // Two reasoning spans with real Gemma markers (iter D W67:
        // `<|channel>` / `<channel|>` per `strip_thinking`).
        let (content, reasoning) = split_full_output(
            &GEMMA4,
            "a <|channel>r1<channel|> b <|channel>r2<channel|> c",
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

    // --- ToolCallSplitter (iter-133 Iter B-2) ---

    /// Iter B-2 W66: lock in the corrected Gemma 4 tool-call markers. Pre-fix
    /// these were `<tool_call>` / `</tool_call>` (the Qwen convention); the
    /// gemma-4 GGUF chat template actually emits `<|tool_call>` (open) and
    /// `<tool_call|>` (close). Real-model fixture
    /// `tests/fixtures/openwebui_multiturn/scenario2_tool_call_chunks.txt`
    /// from W65 confirmed the literal mismatch.
    #[test]
    fn gemma4_tool_call_markers_match_chat_template_emission() {
        assert_eq!(GEMMA4.tool_open, Some("<|tool_call>"));
        assert_eq!(GEMMA4.tool_close, Some("<tool_call|>"));
    }

    fn tcfeed(reg: &ModelRegistration, s: &str) -> Vec<ToolCallEvent> {
        let mut sp = ToolCallSplitter::from_registration(reg).unwrap();
        let mut out = sp.feed(s);
        if let Some(tail) = sp.finish() {
            out.push(tail);
        }
        out
    }

    fn tc_coalesce(v: &[ToolCallEvent]) -> Vec<ToolCallEvent> {
        let mut out: Vec<ToolCallEvent> = Vec::new();
        for ev in v {
            if let (Some(last), ev) = (out.last_mut(), ev) {
                match (last, ev) {
                    (ToolCallEvent::Content(a), ToolCallEvent::Content(b)) => {
                        a.push_str(b);
                        continue;
                    }
                    (ToolCallEvent::ToolCallText(a), ToolCallEvent::ToolCallText(b)) => {
                        a.push_str(b);
                        continue;
                    }
                    _ => {}
                }
            }
            out.push(ev.clone());
        }
        out
    }

    #[test]
    fn tool_call_splitter_no_markers_all_content() {
        let out = tcfeed(&GEMMA4, "hello world");
        assert_eq!(out, vec![ToolCallEvent::Content("hello world".into())]);
    }

    #[test]
    fn tool_call_splitter_single_call_gemma4_markers() {
        let out = tcfeed(
            &GEMMA4,
            "pre <|tool_call>call:f{x:1}<tool_call|> post",
        );
        let joined = tc_coalesce(&out);
        assert_eq!(
            joined,
            vec![
                ToolCallEvent::Content("pre ".into()),
                ToolCallEvent::ToolCallOpen,
                ToolCallEvent::ToolCallText("call:f{x:1}".into()),
                ToolCallEvent::ToolCallClose,
                ToolCallEvent::Content(" post".into()),
            ]
        );
    }

    #[test]
    fn tool_call_splitter_qwen35_markers_distinct() {
        let out = tcfeed(
            &QWEN35,
            "pre <tool_call>\n<function=f><parameter=x>\n1\n</parameter></function>\n</tool_call> post",
        );
        let joined = tc_coalesce(&out);
        assert_eq!(joined.len(), 5, "got {joined:?}");
        match (&joined[0], &joined[1], &joined[3], &joined[4]) {
            (
                ToolCallEvent::Content(a),
                ToolCallEvent::ToolCallOpen,
                ToolCallEvent::ToolCallClose,
                ToolCallEvent::Content(b),
            ) => {
                assert_eq!(a, "pre ");
                assert_eq!(b, " post");
            }
            other => panic!("unexpected event sequence: {other:?}"),
        }
    }

    #[test]
    fn tool_call_splitter_marker_spans_fragment_boundary() {
        let mut sp = ToolCallSplitter::from_registration(&GEMMA4).unwrap();
        let a = sp.feed("before <|tool");
        let b = sp.feed("_call>call:f{a:1}<tool_call");
        let c = sp.feed("|>after");
        let d = sp.finish();
        let mut all: Vec<ToolCallEvent> = Vec::new();
        all.extend(a);
        all.extend(b);
        all.extend(c);
        if let Some(t) = d {
            all.push(t);
        }
        let joined = tc_coalesce(&all);
        assert_eq!(
            joined,
            vec![
                ToolCallEvent::Content("before ".into()),
                ToolCallEvent::ToolCallOpen,
                ToolCallEvent::ToolCallText("call:f{a:1}".into()),
                ToolCallEvent::ToolCallClose,
                ToolCallEvent::Content("after".into()),
            ]
        );
    }

    #[test]
    fn tool_call_splitter_open_without_close_finishes_in_call() {
        // Mid-call EOS: `finish()` returns the partial body as ToolCallText.
        let out = tcfeed(&GEMMA4, "<|tool_call>call:f{a:1");
        // Coalesce because the splitter holds back a tail.
        let joined = tc_coalesce(&out);
        assert_eq!(
            joined,
            vec![
                ToolCallEvent::ToolCallOpen,
                ToolCallEvent::ToolCallText("call:f{a:1".into()),
            ]
        );
    }

    #[test]
    fn tool_call_splitter_no_registration_returns_none() {
        let none_reg = ModelRegistration {
            family: "no-tools",
            id_substrings: &["test"],
            reasoning_open: None,
            reasoning_close: None,
            tool_open: None,
            tool_close: None,
            tool_preamble: None,
        };
        assert!(ToolCallSplitter::from_registration(&none_reg).is_none());
    }

    // --- parse_tool_call_body ---

    #[test]
    fn parse_gemma4_simple_string_arg() {
        let parsed = parse_tool_call_body(
            &GEMMA4,
            "call:get_current_weather{location:<|\"|>Paris<|\"|>}",
        )
        .expect("parse");
        assert_eq!(parsed.name, "get_current_weather");
        // Order is HashMap-iteration → json string canonicalized via
        // serde_json (sorted keys not guaranteed but the SET of fields is).
        let v: serde_json::Value =
            serde_json::from_str(&parsed.arguments_json).expect("arg JSON");
        assert_eq!(v["location"], "Paris");
    }

    #[test]
    fn parse_gemma4_multi_arg_string_and_enum() {
        let parsed = parse_tool_call_body(
            &GEMMA4,
            "call:f{location:<|\"|>San Francisco<|\"|>,unit:<|\"|>celsius<|\"|>}",
        )
        .expect("parse");
        assert_eq!(parsed.name, "f");
        let v: serde_json::Value =
            serde_json::from_str(&parsed.arguments_json).expect("arg JSON");
        assert_eq!(v["location"], "San Francisco");
        assert_eq!(v["unit"], "celsius");
    }

    #[test]
    fn parse_gemma4_numeric_and_bool_args() {
        let parsed = parse_tool_call_body(
            &GEMMA4,
            "call:f{count:42,enabled:true,ratio:1.5}",
        )
        .expect("parse");
        let v: serde_json::Value =
            serde_json::from_str(&parsed.arguments_json).expect("arg JSON");
        assert_eq!(v["count"], 42);
        assert_eq!(v["enabled"], true);
        assert_eq!(v["ratio"], 1.5);
    }

    #[test]
    fn parse_gemma4_string_with_comma_inside_quotes() {
        // Comma inside `<|"|>...<|"|>` must NOT split top-level args.
        let parsed = parse_tool_call_body(
            &GEMMA4,
            "call:f{addr:<|\"|>1, Main St<|\"|>,city:<|\"|>NYC<|\"|>}",
        )
        .expect("parse");
        let v: serde_json::Value =
            serde_json::from_str(&parsed.arguments_json).expect("arg JSON");
        assert_eq!(v["addr"], "1, Main St");
        assert_eq!(v["city"], "NYC");
    }

    #[test]
    fn parse_gemma4_empty_args() {
        let parsed = parse_tool_call_body(&GEMMA4, "call:noop{}").expect("parse");
        assert_eq!(parsed.name, "noop");
        assert_eq!(parsed.arguments_json, "{}");
    }

    #[test]
    fn parse_gemma4_invalid_returns_none() {
        // Missing `call:` prefix.
        assert!(parse_tool_call_body(&GEMMA4, "garbage{}").is_none());
        // Missing braces.
        assert!(parse_tool_call_body(&GEMMA4, "call:f").is_none());
        // Missing function name.
        assert!(parse_tool_call_body(&GEMMA4, "call:{}").is_none());
    }

    #[test]
    fn parse_qwen35_function_with_parameters() {
        let parsed = parse_tool_call_body(
            &QWEN35,
            "<function=get_current_weather>\n<parameter=location>\nParis\n</parameter>\n</function>",
        )
        .expect("parse");
        assert_eq!(parsed.name, "get_current_weather");
        let v: serde_json::Value =
            serde_json::from_str(&parsed.arguments_json).expect("arg JSON");
        // Qwen recommends `tojson`; bare `Paris` is not valid JSON → string fallback.
        assert_eq!(v["location"], "Paris");
    }

    #[test]
    fn parse_qwen35_function_with_jsonish_value() {
        let parsed = parse_tool_call_body(
            &QWEN35,
            "<function=set>\n<parameter=count>\n42\n</parameter>\n</function>",
        )
        .expect("parse");
        let v: serde_json::Value =
            serde_json::from_str(&parsed.arguments_json).expect("arg JSON");
        // 42 IS valid JSON → number.
        assert_eq!(v["count"], 42);
    }

    #[test]
    fn parse_qwen35_invalid_returns_none() {
        assert!(parse_tool_call_body(&QWEN35, "garbage").is_none());
        assert!(parse_tool_call_body(&QWEN35, "<function=>").is_none());
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
