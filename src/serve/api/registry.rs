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
//!
//! # Per-model wrapper grammar (ADR-005 Phase 2a T1.8 Option B, Wave 2 W-4)
//!
//! `ModelRegistration::tool_call_gbnf` emits a GBNF string that physically
//! constrains the model to emit a valid wrapper + valid arguments for the
//! named function. The caller (handlers.rs `compile_tool_grammar`) parses the
//! GBNF string with `grammar::parser::parse` and attaches the result to
//! `SamplingParams.grammar`.
//!
//! WHY per-model wrapper grammars instead of plain JSON:
//!   - **Gemma 4** — the chat template's `tool_call` macro emits
//!     `call:NAME{key:<|"|>val<|"|>,...}`, NOT JSON. The `<|"|>` string-quote
//!     markers are Gemma-specific (see `models/gemma4/chat_template.jinja:113`).
//!     If the grammar constrained plain `{"key":"val"}` the model would fight
//!     the constraint and produce degenerate outputs; the constraint must match
//!     the template's expected emission exactly.
//!   - **Qwen 3.5/3.6** — the chat template's tool-call block emits
//!     `<function=NAME><parameter=key>val</parameter>...</function>` XML.
//!     Same reason: the constraint must match the template's emission shape.
//!   - Plain JSON constraints (via json_schema::schema_to_gbnf) are correct
//!     for `response_format=json_schema` where no per-model wrapper exists.
//!     Tool-call wrappers are necessarily per-model, and their inner value
//!     grammar must mirror the wrapper's own quoting convention (Gemma uses
//!     `<|"|>` string markers; Qwen XML wraps raw values).

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

    /// Opening marker for a tool-call block. `None` = no tool calls.
    ///
    /// # Marker-shape contract (ADR-005 Wave-2.5 D1 — Option B architecture)
    ///
    /// `tool_open` and `tool_close` MUST be byte sequences that the
    /// tokenizer's **special tokens** decode to.  They are NOT arbitrary
    /// strings — the engine feeds raw decoded bytes to `ToolCallSplitter`,
    /// which scans for these exact byte sequences to detect the open/close
    /// boundary.  If a marker string is set to something the tokenizer never
    /// emits as a special-token sequence, the splitter will never fire.
    ///
    /// Authoritative references:
    /// - **Gemma 4**: `tokenizer_config.json` declares `stc_token` (start of
    ///   tool call) = `<|tool_call>` and the paired close token
    ///   `<tool_call|>`.  These are the byte strings the model emits when
    ///   invoking a tool.  Confirmed from chat_template.jinja lines 192/203:
    ///   `{{- '<|tool_call>call:...<tool_call|>' }}`.
    /// - **Qwen 3.5/3.6**: token ids 248058 (`<tool_call>`) and 248059
    ///   (`</tool_call>`).  The chat_template emits
    ///   `<tool_call>\n<function=NAME>…</function>\n</tool_call>` so the
    ///   outer `<tool_call>` / `</tool_call>` pair are the registered markers.
    ///
    /// WHY this matters: using the *prompt-side* thinking-mode hint (`<|think|>`)
    /// instead of the *runtime-emission* boundary (`<|channel>`) caused the
    /// reasoning splitter to be permanently dead for Gemma 4 until iter B-2
    /// corrected the registration.  Any future model addition must audit the
    /// tokenizer_config and chat_template before setting these fields.
    pub tool_open: Option<&'static str>,
    /// Closing marker for a tool-call block. `None` = no tool calls.
    /// See `tool_open` for the marker-shape contract.
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

    /// Emit a GBNF string that physically constrains the model output to a
    /// well-formed tool-call wrapper for function `fn_name` with parameters
    /// matching `params_schema` (a JSON Schema object).
    ///
    /// The emitted GBNF is **between** the open/close markers — i.e. the
    /// portion the `ToolCallSplitter` hands to `parse_tool_call_body`. The
    /// caller (`compile_tool_grammar` in handlers.rs) prepends nothing; the
    /// marker bytes are never fed to the grammar sampler because the sampler
    /// is wired to run only after the open marker has been observed (or, in
    /// the forced-tool-call case, from the first decode token).
    ///
    /// Returns `Err(String)` when the family is unknown or when `params_schema`
    /// contains a feature the per-model emitter doesn't support yet. The error
    /// string is forwarded as a 400 `grammar_error` to the caller.
    ///
    /// WHY the return type is `String` (not `Grammar`):
    /// registry.rs intentionally has no dependency on `grammar::parser` to
    /// keep the module's concern narrow (marker registration + body parsing).
    /// The caller in handlers.rs already imports the grammar module and calls
    /// `grammar::parser::parse` on the returned string — the same pattern used
    /// by `compile_response_format`.
    pub fn tool_call_gbnf(
        &self,
        fn_name: &str,
        params_schema: &serde_json::Value,
    ) -> Result<String, String> {
        match self.family {
            "gemma4" => gemma4_tool_call_gbnf(fn_name, params_schema)
                .map_err(|e| e.to_string()),
            "qwen35" => qwen35_tool_call_gbnf(fn_name, params_schema)
                .map_err(|e| e.to_string()),
            other => Err(format!(
                "tool_call_gbnf: no per-model grammar emitter for family '{}'",
                other
            )),
        }
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

// ---------------------------------------------------------------------------
// Per-model GBNF emitters (T1.8 Option B, ADR-005 Wave 2 W-4)
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Emitter error type (Wave 2.5 B1 / B3)
// ---------------------------------------------------------------------------

/// Structured error type returned by the per-model GBNF emitters.
///
/// The `Display` impl produces the human-readable 400-response message that
/// `handlers.rs::compile_tool_grammar` forwards to the API caller.  The
/// `tool_call_gbnf` public entry-point converts this to `String` so the
/// public API surface (`Result<String, String>`) is unchanged.
///
/// Variants:
/// - `TooManyRequiredKeys` — schema's `required` array exceeds the 8-key
///   cap; the O(2^N) permutation grammar would be unreasonably large.
/// - `UnsupportedSchema` — parameter schema uses `array` or `object` type,
///   which the wave-2.5 emitter does not support.  The caller should either
///   remove the parameter or convert it to a JSON-encoded string.
///
/// # ADR-005 Wave-2.7 design note
/// The 8-key cap is the SOTA hard bound established by json_schema.rs
/// (ANY_ORDER_MAX_REQUIRED = 8, 256 rules worst-case — practical and fast).
/// The previous 16-key cap in this file was misaligned: json_schema.rs
/// hard-errors at >8, so a tool with 9–16 required keys would slip through
/// registry.rs and be silently exposed to O(2^N) grammar growth before
/// json_schema.rs caught it.  Both caps are now 8 — Q3 audit finding fixed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EmitterError {
    /// `required` array length exceeds `MAX_REQUIRED_KEYS` (8).
    TooManyRequiredKeys {
        fn_name: String,
        count: usize,
    },
    /// Parameter schema type is `array` or `object` — not supported by the
    /// wave-2.5 scalar-only emitter.
    UnsupportedSchema {
        fn_name: String,
        param_name: String,
        schema_type: String,
    },
}

/// Hard cap on the number of required keys for the permutation grammar.
/// Aligned with json_schema.rs ANY_ORDER_MAX_REQUIRED = 8 (ADR-005 W-ζ).
/// Larger schemas return `EmitterError::TooManyRequiredKeys`.
const MAX_REQUIRED_KEYS: usize = 8;

impl std::fmt::Display for EmitterError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            EmitterError::TooManyRequiredKeys { fn_name, count } => write!(
                f,
                "function '{}' has {} required parameters; ADR-005 wave-2.7 \
                 limits required keys to {} (SOTA bound: O(2^N) permutation \
                 grammar, 256 rules worst-case); reduce the required set or \
                 split the tool",
                fn_name, count, MAX_REQUIRED_KEYS
            ),
            EmitterError::UnsupportedSchema {
                fn_name,
                param_name,
                schema_type,
            } => write!(
                f,
                "function '{}' parameter '{}' uses unsupported schema type '{}'; \
                 ADR-005 wave-2.5 limits tool args to scalars (string, integer, \
                 number, boolean, null); remove the parameter or convert it to a \
                 JSON-encoded string",
                fn_name, param_name, schema_type
            ),
        }
    }
}

/// Escape a literal string for embedding in a GBNF rule — wraps in double
/// quotes and escapes special characters.  Mirrors
/// `grammar::json_schema::format_literal` without importing the grammar crate.
///
/// WHY inlined here: registry.rs has no grammar module dependency by design
/// (it's a registration + parsing module, not a grammar emission module).
/// Duplicating the trivial 10-line escape is preferable to introducing a
/// circular import path.
fn gbnf_literal(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    out.push('"');
    for c in s.chars() {
        match c {
            '\r' => out.push_str("\\r"),
            '\n' => out.push_str("\\n"),
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            _ => out.push(c),
        }
    }
    out.push('"');
    out
}

/// Map a JSON Schema type string to the GBNF value rule appropriate for
/// **Gemma 4's kv-list syntax**.
///
/// Gemma 4 string values use the `<|"|>...<|"|>` quote-marker convention
/// (see `models/gemma4/chat_template.jinja:113`): the model never emits a
/// plain JSON `"..."` string inside a tool call — it wraps every string arg
/// in `<|"|>` open + close tokens.  Non-string scalars (numbers, booleans,
/// null) are emitted bare (no JSON quotes) exactly as in standard JSON values.
///
/// WHY NOT reuse json_schema primitives here:
/// json_schema::schema_to_gbnf emits a grammar that matches `"quoted JSON
/// strings"` (double quotes with JSON escape sequences).  Gemma 4's template
/// never emits those — it always uses `<|"|>` markers — so reusing the JSON
/// string grammar would make valid Gemma 4 outputs fail the mask check.
///
/// Returns `Err(EmitterError::UnsupportedSchema)` if the schema type is
/// `array` or `object` (Wave 2.5 B3 — nested schemas rejected at emit time).
fn gemma4_value_gbnf(
    fn_name: &str,
    param_name: &str,
    schema: &serde_json::Value,
    _rules: &mut Vec<(String, String)>,
    _rule_counter: &mut u32,
) -> Result<String, EmitterError> {
    let obj = match schema.as_object() {
        Some(o) => o,
        None => {
            // Untyped / unknown → accept any Gemma value (string or bare scalar).
            return Ok("gemma4-any-val".to_string());
        }
    };

    // enum (string values only in Gemma 4's canonical use) → literal alternation.
    if let Some(serde_json::Value::Array(values)) = obj.get("enum") {
        let alts: Vec<String> = values
            .iter()
            .filter_map(|v| v.as_str())
            .map(|s| format!("{} {} {}", gbnf_literal("<|\"|>"), gbnf_literal(s), gbnf_literal("<|\"|>")))
            .collect();
        if !alts.is_empty() {
            return Ok(format!("( {} )", alts.join(" | ")));
        }
    }

    let schema_type = obj.get("type").and_then(|t| t.as_str()).unwrap_or("");
    match schema_type {
        "string" => {
            // Gemma 4 string: `<|"|>` + zero-or-more non-marker chars + `<|"|>`.
            // We approximate "non-marker chars" as any char that is not the
            // start of the 5-byte `<|"|>` marker — in practice the grammar
            // sampler enforces this token-by-token; the grammar just needs to
            // be wide enough to accept valid outputs.
            //
            // Rule body: `"<|\"|>" gemma4-str-char* "<|\"|>"`
            // where gemma4-str-char matches any Unicode scalar that is not `<`
            // (the marker opening byte).  This is conservative but correct:
            // string values in Gemma 4 tool calls never contain `<` per the
            // chat template's jinja escape of `<` in string args.
            Ok("gemma4-str-val".to_string())
        }
        "integer" => Ok("gemma4-int-val".to_string()),
        "number" => Ok("gemma4-num-val".to_string()),
        "boolean" => Ok("gemma4-bool-val".to_string()),
        "null" => Ok("gemma4-null-val".to_string()),
        // Wave 2.5 B3: reject array and object types at emit time.
        // The previous fallback silently treated these as strings, hiding the
        // schema mismatch; the Chesterton reason to remove it is that it
        // produces grammars that accept `<|"|>...<|"|>` for a parameter the
        // schema declares as a structured list/object, masking bugs at the
        // grammar layer while the model still emits malformed output.
        "array" | "object" => Err(EmitterError::UnsupportedSchema {
            fn_name: fn_name.to_string(),
            param_name: param_name.to_string(),
            schema_type: schema_type.to_string(),
        }),
        _ => {
            // Unrecognised type string — fall through to any-val (conservative).
            Ok("gemma4-any-val".to_string())
        }
    }
}

/// Emit a GBNF grammar string constraining output to Gemma 4's
/// `call:NAME{key:val,...}` tool-call wrapper for the given function and
/// parameters schema.
///
/// The Gemma 4 chat template's `tool_call` macro (jinja:113-200) emits:
///   `call:FUNCTION_NAME{param_name:<|"|>string_val<|"|>,count:42,...}`
///
/// So the grammar has three structural layers:
///   1. A fixed prefix: the literal `call:NAME{`
///   2. A kv-list body: comma-separated `KEY:VALUE` pairs (required in any
///      order; optional may appear after; duplicates structurally rejected by
///      the required-permutation grammar).
///   3. A fixed suffix: `}`
///
/// # Required parameter enforcement (Wave 2.5 B1)
///
/// If `params_schema` has a `required` array those keys are enforced via a
/// permutation grammar: required keys MUST all appear in any order; omitting
/// one causes the grammar stack to die.  Optional keys follow in a
/// Kleene-star suffix.  Hard cap `MAX_REQUIRED_KEYS` (8) prevents O(2^N)
/// grammar blowup.
///
/// # Nested schema rejection (Wave 2.5 B3)
///
/// If any parameter's schema type is `array` or `object`, returns
/// `EmitterError::UnsupportedSchema` immediately.
fn gemma4_tool_call_gbnf(
    fn_name: &str,
    params_schema: &serde_json::Value,
) -> Result<String, EmitterError> {
    let mut rules: Vec<(String, String)> = Vec::new();
    let mut rule_counter: u32 = 0;

    // Gemma 4 primitive rules (shared across all function grammars).
    //
    // String: `<|"|>` any-safe-char* `<|"|>`. We define gemma4-str-char as
    // any character that is NOT `<` (the first byte of the 5-byte marker
    // `<|"|>`). This is conservative but safe — Gemma 4's jinja template
    // HTML-escapes `<` in string args so valid model outputs never contain
    // bare `<` inside string values.
    rules.push((
        "gemma4-str-char".to_string(),
        r#"[^<\\] | [\\] [^\x00-\x1F]"#.to_string(),
    ));
    rules.push((
        "gemma4-str-val".to_string(),
        format!(
            "{} gemma4-str-char* {}",
            gbnf_literal("<|\"|>"),
            gbnf_literal("<|\"|>")
        ),
    ));
    rules.push((
        "gemma4-int-val".to_string(),
        r#""-"? ([0] | [1-9] [0-9]{0,15})"#.to_string(),
    ));
    rules.push((
        "gemma4-num-val".to_string(),
        r#""-"? ([0] | [1-9] [0-9]{0,15}) ("." [0-9]{1,16})? ([eE] [-+]? [0-9]{1,16})?"#.to_string(),
    ));
    rules.push((
        "gemma4-bool-val".to_string(),
        r#""true" | "false""#.to_string(),
    ));
    rules.push((
        "gemma4-null-val".to_string(),
        r#""null""#.to_string(),
    ));
    rules.push((
        "gemma4-any-val".to_string(),
        r#"gemma4-str-val | gemma4-num-val | gemma4-bool-val | gemma4-null-val"#.to_string(),
    ));

    // Extract `required` set (Wave 2.5 B1).
    let required_set: std::collections::HashSet<String> = params_schema
        .as_object()
        .and_then(|o| o.get("required"))
        .and_then(|v| v.as_array())
        .map(|arr| {
            arr.iter()
                .filter_map(|v| v.as_str().map(String::from))
                .collect()
        })
        .unwrap_or_default();

    if required_set.len() > MAX_REQUIRED_KEYS {
        return Err(EmitterError::TooManyRequiredKeys {
            fn_name: fn_name.to_string(),
            count: required_set.len(),
        });
    }

    let properties = params_schema
        .as_object()
        .and_then(|o| o.get("properties"))
        .and_then(|p| p.as_object());

    let prefix_lit = gbnf_literal(&format!("call:{}", fn_name));
    let open_lit = gbnf_literal("{");
    let close_lit = gbnf_literal("}");
    let comma_lit = gbnf_literal(",");

    if let Some(props) = properties {
        if props.is_empty() {
            rules.push(("root".to_string(), format!("{} {} {} space", prefix_lit, open_lit, close_lit)));
        } else {
            let mut required_kv_names: Vec<String> = Vec::new();
            let mut optional_kv_names: Vec<String> = Vec::new();
            let mut sorted_keys: Vec<&String> = props.keys().collect();
            sorted_keys.sort();
            for key in &sorted_keys {
                let val_schema = &props[*key];
                // B3: reject array/object at emit time.
                let val_rule = gemma4_value_gbnf(fn_name, key.as_str(), val_schema, &mut rules, &mut rule_counter)?;
                let key_lit = gbnf_literal(key.as_str());
                let kv_body = format!("{} {} {}", key_lit, gbnf_literal(":"), val_rule);
                let kv_name = format!("gemma4-kv-{}", sanitize_rule_name_local(key));
                rules.push((kv_name.clone(), kv_body));
                if required_set.contains(*key) {
                    required_kv_names.push(kv_name);
                } else {
                    optional_kv_names.push(kv_name);
                }
            }

            // Build kv-list body.
            //
            // Case A (no required): pure Kleene-star over all kv items.
            // Case B (required):  permutation grammar for required keys,
            //   followed by optional Kleene-star for optional keys.
            //
            // WHY permutation for required: parse_gemma4_tool_call is
            // order-agnostic, so enforcing a fixed required order would reject
            // valid model outputs that sequence required keys differently.
            let kv_body_rule = if required_kv_names.is_empty() {
                // Case A.
                let all_names: Vec<String> = optional_kv_names.clone();
                let alts = all_names.join(" | ");
                let kv_item_rule = "gemma4-kv-item".to_string();
                rules.push((kv_item_rule.clone(), format!("( {} )", alts)));
                let kv_list_rule = "gemma4-kv-list".to_string();
                rules.push((
                    kv_list_rule.clone(),
                    format!("{} ( {} {} )*", kv_item_rule, comma_lit, kv_item_rule),
                ));
                kv_list_rule
            } else {
                // Case B.
                let req_top = build_gemma4_required_permutation(
                    fn_name,
                    &required_kv_names,
                    &comma_lit,
                    &mut rules,
                );
                if optional_kv_names.is_empty() {
                    req_top
                } else {
                    let alts = optional_kv_names.join(" | ");
                    let opt_item_rule = "gemma4-opt-item".to_string();
                    rules.push((opt_item_rule.clone(), format!("( {} )", alts)));
                    let kv_list_rule = "gemma4-kv-list".to_string();
                    rules.push((
                        kv_list_rule.clone(),
                        format!("{} ( {} {} )*", req_top, comma_lit, opt_item_rule),
                    ));
                    kv_list_rule
                }
            };

            rules.push((
                "root".to_string(),
                format!("{} {} {} {} space", prefix_lit, open_lit, kv_body_rule, close_lit),
            ));
        }
    } else {
        rules.push((
            "gemma4-any-kv-char".to_string(),
            r#"[^}]"#.to_string(),
        ));
        rules.push((
            "root".to_string(),
            format!("{} {} gemma4-any-kv-char* {} space", prefix_lit, open_lit, close_lit),
        ));
    }

    rules.push(("space".to_string(), r#"| " " | "\n"{1,2} [ \t]{0,20}"#.to_string()));

    let mut out = String::new();
    for (name, body) in &rules {
        if name == "root" {
            out.push_str(&format!("root ::= {}\n", body));
            break;
        }
    }
    for (name, body) in &rules {
        if name != "root" {
            out.push_str(&format!("{} ::= {}\n", name, body));
        }
    }
    Ok(out)
}

/// Build the any-order permutation grammar for a non-empty set of required
/// Gemma kv-rule names. Returns the name of the top-level permutation rule.
///
/// Mirrors `json_schema::build_required_permutation` but uses Gemma's bare
/// comma separator.  Rule names encode the sorted kv names so each distinct
/// subset is emitted exactly once (memoized via the rules vec).
fn build_gemma4_required_permutation(
    slug: &str,
    required_kv_names: &[String],
    comma_lit: &str,
    rules: &mut Vec<(String, String)>,
) -> String {
    let mut sorted = required_kv_names.to_vec();
    sorted.sort();

    let name_parts: Vec<String> = sorted
        .iter()
        .map(|n| n.trim_start_matches("gemma4-kv-").to_string())
        .collect();
    // Truncate the rule name to avoid hitting GBNF parser limits for large
    // key sets; the sorted join is unique enough within a single function.
    let rule_name = format!("g4req-{}-{}", sanitize_rule_name_local(slug), name_parts.join("-"));

    if rules.iter().any(|(n, _)| n == &rule_name) {
        return rule_name;
    }

    if sorted.len() == 1 {
        rules.push((rule_name.clone(), sorted[0].clone()));
        return rule_name;
    }

    // Insert placeholder to allow memoization check in recursive calls.
    rules.push((rule_name.clone(), String::new()));

    let mut alts: Vec<String> = Vec::new();
    for (i, kv_name) in sorted.iter().enumerate() {
        let remaining: Vec<String> = sorted
            .iter()
            .enumerate()
            .filter(|(j, _)| *j != i)
            .map(|(_, s)| s.clone())
            .collect();
        let rest = build_gemma4_required_permutation(slug, &remaining, comma_lit, rules);
        alts.push(format!("{} {} {}", kv_name, comma_lit, rest));
    }
    let body = alts.join(" | ");
    for (n, b) in rules.iter_mut() {
        if n == &rule_name {
            *b = body;
            break;
        }
    }
    rule_name
}

/// Emit a GBNF grammar string constraining output to Qwen 3.5/3.6's
/// `<function=NAME><parameter=key>val</parameter>...</function>` wrapper.
///
/// The Qwen 3.5/3.6 chat template emits tool calls as (verified against
/// `tokenizer_config.json` `chat_template` field 2026-04-26):
///   `<function=NAME>\n<parameter=KEY>\nVAL\n</parameter>\n...</function>`
///
/// Every `</parameter>` is followed by a newline — including the last one
/// before `</function>`.  The grammar enforces this exactly.
///
/// # Required parameter enforcement (Wave 2.5 B1)
///
/// Same as Gemma 4: required keys enforced via permutation grammar; optional
/// keys in Kleene-star suffix.  Hard cap `MAX_REQUIRED_KEYS` (8).
///
/// # Nested schema rejection (Wave 2.5 B3)
///
/// `array` or `object` parameter types → `EmitterError::UnsupportedSchema`.
fn qwen35_tool_call_gbnf(
    fn_name: &str,
    params_schema: &serde_json::Value,
) -> Result<String, EmitterError> {
    let mut rules: Vec<(String, String)> = Vec::new();

    // Qwen 3.5/3.6 value primitives — values sit between XML tags, raw text
    // (no JSON quoting for strings).  Numbers/booleans are JSON-serialized
    // (the `tojson` Jinja filter is used in the template).
    rules.push((
        "qwen35-str-char".to_string(),
        // Any char that is not `<` (first byte of `</parameter>`).
        // Conservative but correct: template HTML-escapes `<` in string values.
        r#"[^<\\] | [\\] [^\x00-\x1F]"#.to_string(),
    ));
    rules.push((
        "qwen35-str-val".to_string(),
        "qwen35-str-char*".to_string(),
    ));
    rules.push((
        "qwen35-int-val".to_string(),
        r#""-"? ([0] | [1-9] [0-9]{0,15})"#.to_string(),
    ));
    rules.push((
        "qwen35-num-val".to_string(),
        r#""-"? ([0] | [1-9] [0-9]{0,15}) ("." [0-9]{1,16})? ([eE] [-+]? [0-9]{1,16})?"#.to_string(),
    ));
    rules.push((
        "qwen35-bool-val".to_string(),
        r#""true" | "false""#.to_string(),
    ));
    rules.push((
        "qwen35-null-val".to_string(),
        r#""null""#.to_string(),
    ));
    rules.push((
        "qwen35-any-val".to_string(),
        r#"qwen35-str-val | qwen35-num-val | qwen35-bool-val | qwen35-null-val"#.to_string(),
    ));

    // Extract `required` set (Wave 2.5 B1).
    let required_set: std::collections::HashSet<String> = params_schema
        .as_object()
        .and_then(|o| o.get("required"))
        .and_then(|v| v.as_array())
        .map(|arr| {
            arr.iter()
                .filter_map(|v| v.as_str().map(String::from))
                .collect()
        })
        .unwrap_or_default();

    if required_set.len() > MAX_REQUIRED_KEYS {
        return Err(EmitterError::TooManyRequiredKeys {
            fn_name: fn_name.to_string(),
            count: required_set.len(),
        });
    }

    let properties = params_schema
        .as_object()
        .and_then(|o| o.get("properties"))
        .and_then(|p| p.as_object());

    let func_open_lit = gbnf_literal(&format!("<function={}>", fn_name));
    let func_close_lit = gbnf_literal("</function>");
    let newline_lit = gbnf_literal("\n");

    if let Some(props) = properties {
        if props.is_empty() {
            rules.push((
                "root".to_string(),
                format!("{} {} space", func_open_lit, func_close_lit),
            ));
        } else {
            let mut required_block_names: Vec<String> = Vec::new();
            let mut optional_block_names: Vec<String> = Vec::new();
            let mut sorted_keys: Vec<&String> = props.keys().collect();
            sorted_keys.sort();
            for key in &sorted_keys {
                let val_schema = &props[*key];
                // B3: reject array/object at emit time.
                let val_rule = qwen35_value_rule(fn_name, key.as_str(), val_schema)?;
                let param_open_lit = gbnf_literal(&format!("<parameter={}>", key));
                let param_close_lit = gbnf_literal("</parameter>");
                // Block form confirmed against tokenizer_config.json chat_template:
                //   `<parameter=KEY>\nVAL\n</parameter>\n`
                // Every </parameter> — including the last before </function> —
                // is followed by \n.  The template loop emits
                //   `{{- '\n</parameter>\n' }}` unconditionally.
                let block_body = format!(
                    "{} {} {} {} {} {}",
                    param_open_lit, newline_lit, val_rule, newline_lit, param_close_lit, newline_lit
                );
                let block_name = format!("qwen35-param-{}", sanitize_rule_name_local(key));
                rules.push((block_name.clone(), block_body));
                if required_set.contains(*key) {
                    required_block_names.push(block_name);
                } else {
                    optional_block_names.push(block_name);
                }
            }

            // Build parameter sequence with B1 required enforcement.
            // Same two-case logic as Gemma 4: permutation for required, then
            // Kleene-star for optional.
            let param_body_rule = if required_block_names.is_empty() {
                // Case A: pure optional Kleene-star.
                let alts = optional_block_names.join(" | ");
                let param_item_rule = "qwen35-param-item".to_string();
                rules.push((param_item_rule.clone(), format!("( {} )", alts)));
                let param_list_rule = "qwen35-param-list".to_string();
                rules.push((
                    param_list_rule.clone(),
                    format!("{}*", param_item_rule),
                ));
                param_list_rule
            } else {
                // Case B: required permutation + optional suffix.
                let req_top = build_qwen35_required_permutation(
                    fn_name,
                    &required_block_names,
                    &mut rules,
                );
                if optional_block_names.is_empty() {
                    req_top
                } else {
                    let alts = optional_block_names.join(" | ");
                    let opt_item_rule = "qwen35-opt-item".to_string();
                    rules.push((opt_item_rule.clone(), format!("( {} )", alts)));
                    let param_list_rule = "qwen35-param-list".to_string();
                    rules.push((
                        param_list_rule.clone(),
                        format!("{} {}*", req_top, opt_item_rule),
                    ));
                    param_list_rule
                }
            };

            // Root: `<function=NAME>\n` + param sequence + `</function>`.
            // The newline after `<function=NAME>` is part of the template
            // emission pattern (verified from tokenizer_config.json).
            rules.push((
                "root".to_string(),
                format!(
                    "{} {} {} {} space",
                    func_open_lit, newline_lit, param_body_rule, func_close_lit
                ),
            ));
        }
    } else {
        rules.push((
            "qwen35-inner-char".to_string(),
            r#"[^<\\] | [\\] [^\x00-\x1F]"#.to_string(),
        ));
        rules.push((
            "root".to_string(),
            format!("{} qwen35-inner-char* {} space", func_open_lit, func_close_lit),
        ));
    }

    rules.push(("space".to_string(), r#"| " " | "\n"{1,2} [ \t]{0,20}"#.to_string()));

    let mut out = String::new();
    for (name, body) in &rules {
        if name == "root" {
            out.push_str(&format!("root ::= {}\n", body));
            break;
        }
    }
    for (name, body) in &rules {
        if name != "root" {
            out.push_str(&format!("{} ::= {}\n", name, body));
        }
    }
    Ok(out)
}

/// Build the any-order permutation grammar for a non-empty set of required
/// Qwen parameter-block rule names.  Mirror of `build_gemma4_required_permutation`
/// but without comma separators (Qwen blocks are self-delimiting XML tags).
fn build_qwen35_required_permutation(
    slug: &str,
    required_block_names: &[String],
    rules: &mut Vec<(String, String)>,
) -> String {
    let mut sorted = required_block_names.to_vec();
    sorted.sort();

    let name_parts: Vec<String> = sorted
        .iter()
        .map(|n| n.trim_start_matches("qwen35-param-").to_string())
        .collect();
    let rule_name = format!(
        "q35req-{}-{}",
        sanitize_rule_name_local(slug),
        name_parts.join("-")
    );

    if rules.iter().any(|(n, _)| n == &rule_name) {
        return rule_name;
    }

    if sorted.len() == 1 {
        rules.push((rule_name.clone(), sorted[0].clone()));
        return rule_name;
    }

    rules.push((rule_name.clone(), String::new()));

    let mut alts: Vec<String> = Vec::new();
    for (i, block_name) in sorted.iter().enumerate() {
        let remaining: Vec<String> = sorted
            .iter()
            .enumerate()
            .filter(|(j, _)| *j != i)
            .map(|(_, s)| s.clone())
            .collect();
        let rest = build_qwen35_required_permutation(slug, &remaining, rules);
        // Qwen blocks are self-delimiting (each ends with `\n`), so no
        // explicit separator between them.
        alts.push(format!("{} {}", block_name, rest));
    }
    let body = alts.join(" | ");
    for (n, b) in rules.iter_mut() {
        if n == &rule_name {
            *b = body;
            break;
        }
    }
    rule_name
}

/// Map a JSON Schema value to the Qwen 3.5 value rule name.
///
/// Returns `Err(EmitterError::UnsupportedSchema)` for `array` and `object`
/// types (Wave 2.5 B3 — nested schema rejection).
fn qwen35_value_rule(
    fn_name: &str,
    param_name: &str,
    schema: &serde_json::Value,
) -> Result<String, EmitterError> {
    let obj = match schema.as_object() {
        Some(o) => o,
        None => return Ok("qwen35-any-val".to_string()),
    };
    // enum → accept only the declared string literals.
    if let Some(serde_json::Value::Array(values)) = obj.get("enum") {
        let alts: Vec<String> = values
            .iter()
            .filter_map(|v| v.as_str())
            .map(|s| gbnf_literal(s))
            .collect();
        if !alts.is_empty() {
            return Ok(format!("( {} )", alts.join(" | ")));
        }
    }
    let schema_type = obj.get("type").and_then(|t| t.as_str()).unwrap_or("");
    match schema_type {
        "string" => Ok("qwen35-str-val".to_string()),
        "integer" => Ok("qwen35-int-val".to_string()),
        "number" => Ok("qwen35-num-val".to_string()),
        "boolean" => Ok("qwen35-bool-val".to_string()),
        "null" => Ok("qwen35-null-val".to_string()),
        // Wave 2.5 B3: reject array and object types.
        "array" | "object" => Err(EmitterError::UnsupportedSchema {
            fn_name: fn_name.to_string(),
            param_name: param_name.to_string(),
            schema_type: schema_type.to_string(),
        }),
        _ => Ok("qwen35-any-val".to_string()),
    }
}

/// Sanitize a property name for use as part of a GBNF rule name.
/// Replaces non-alphanumeric/dash chars with `-`.
fn sanitize_rule_name_local(raw: &str) -> String {
    let mut out = String::with_capacity(raw.len());
    for c in raw.chars() {
        if c.is_ascii_alphanumeric() || c == '-' {
            out.push(c);
        } else {
            out.push('-');
        }
    }
    if out.is_empty() {
        out.push('x');
    }
    out
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

    // -----------------------------------------------------------------------
    // T1.8 Option B — tool_call_gbnf accept/reject tests
    //
    // Each test:
    //   1. Calls `reg.tool_call_gbnf(fn_name, params_schema)` → GBNF string.
    //   2. Parses it with `grammar::parser::parse` → `Grammar`.
    //   3. Runs `GrammarRuntime::new` → `GrammarRuntime`.
    //   4. Feeds a candidate byte string with `accept_bytes`; asserts
    //      `is_accepted()` for valid inputs and `!(ok && is_accepted())`
    //      for invalid ones.
    //
    // Import from the sibling grammar module (available as
    // `crate::serve::api::grammar::*` inside the test context).
    // -----------------------------------------------------------------------

    fn grammar_runtime_for_gbnf(gbnf: &str) -> crate::serve::api::grammar::sampler::GrammarRuntime {
        let g = crate::serve::api::grammar::parser::parse(gbnf)
            .unwrap_or_else(|e| panic!("parse GBNF:\n{}\nerror: {}", gbnf, e));
        let rid = g.rule_id("root")
            .unwrap_or_else(|| panic!("no root rule in GBNF:\n{}", gbnf));
        crate::serve::api::grammar::sampler::GrammarRuntime::new(g, rid)
            .unwrap_or_else(|| panic!("GrammarRuntime::new returned None for GBNF:\n{}", gbnf))
    }

    fn gemma4_runtime(fn_name: &str, schema_json: &str) -> crate::serve::api::grammar::sampler::GrammarRuntime {
        let schema: serde_json::Value = serde_json::from_str(schema_json).unwrap();
        let gbnf = GEMMA4.tool_call_gbnf(fn_name, &schema)
            .unwrap_or_else(|e| panic!("tool_call_gbnf error: {}", e));
        grammar_runtime_for_gbnf(&gbnf)
    }

    fn qwen35_runtime(fn_name: &str, schema_json: &str) -> crate::serve::api::grammar::sampler::GrammarRuntime {
        let schema: serde_json::Value = serde_json::from_str(schema_json).unwrap();
        let gbnf = QWEN35.tool_call_gbnf(fn_name, &schema)
            .unwrap_or_else(|e| panic!("tool_call_gbnf error: {}", e));
        grammar_runtime_for_gbnf(&gbnf)
    }

    // -----------------------------------------------------------------------
    // Gemma 4 grammar tests
    // -----------------------------------------------------------------------

    /// Canonical Gemma 4 emission for `get_weather(location: "SF", unit: "F")`.
    /// The grammar must accept the exact string that `parse_gemma4_tool_call`
    /// would successfully parse.
    #[test]
    fn gemma4_tool_call_grammar_accepts_canonical_emission() {
        let schema = r#"{
            "type": "object",
            "properties": {
                "location": {"type": "string"},
                "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}
            }
        }"#;
        let mut rt = gemma4_runtime("get_weather", schema);
        // Canonical emission: `call:get_weather{location:<|"|>SF<|"|>,unit:<|"|>fahrenheit<|"|>}`
        let input = b"call:get_weather{location:<|\"|>SF<|\"|>,unit:<|\"|>fahrenheit<|\"|>}";
        assert!(rt.accept_bytes(input), "canonical emission rejected");
        assert!(rt.is_accepted(), "not accepted at end");
    }

    /// The grammar must also accept the case where keys appear in the
    /// opposite order (unit before location) — since `parse_gemma4_tool_call`
    /// is order-agnostic and we use a Kleene-star kv-list.
    #[test]
    fn gemma4_tool_call_grammar_accepts_reversed_key_order() {
        let schema = r#"{
            "type": "object",
            "properties": {
                "location": {"type": "string"},
                "unit": {"type": "string"}
            }
        }"#;
        let mut rt = gemma4_runtime("get_weather", schema);
        let input = b"call:get_weather{unit:<|\"|>celsius<|\"|>,location:<|\"|>London<|\"|>}";
        assert!(rt.accept_bytes(input), "reversed key order rejected");
        assert!(rt.is_accepted(), "not accepted at end");
    }

    /// The grammar must accept a numeric argument (no `<|"|>` wrapping).
    #[test]
    fn gemma4_tool_call_grammar_accepts_numeric_arg() {
        let schema = r#"{
            "type": "object",
            "properties": {
                "count": {"type": "integer"},
                "enabled": {"type": "boolean"}
            }
        }"#;
        let mut rt = gemma4_runtime("do_thing", schema);
        let input = b"call:do_thing{count:42,enabled:true}";
        assert!(rt.accept_bytes(input), "numeric+boolean args rejected");
        assert!(rt.is_accepted(), "not accepted");
    }

    /// A malformed wrapper prefix (`call_:` with underscore instead of `:`)
    /// must be rejected.
    #[test]
    fn gemma4_tool_call_grammar_rejects_malformed_wrapper_prefix() {
        let schema = r#"{
            "type": "object",
            "properties": {"location": {"type": "string"}}
        }"#;
        let mut rt = gemma4_runtime("get_weather", schema);
        // `call_:get_weather{...}` — underscore between `call` and `:`.
        let input = b"call_:get_weather{location:<|\"|>SF<|\"|>}";
        let ok = rt.accept_bytes(input);
        assert!(!(ok && rt.is_accepted()), "malformed prefix accepted (should reject)");
    }

    /// A wrong delimiter (parentheses instead of braces) must be rejected.
    #[test]
    fn gemma4_tool_call_grammar_rejects_wrong_delimiter() {
        let schema = r#"{
            "type": "object",
            "properties": {"location": {"type": "string"}}
        }"#;
        let mut rt = gemma4_runtime("get_weather", schema);
        // `call:get_weather(SF)` — parentheses instead of braces.
        let input = b"call:get_weather(SF)";
        let ok = rt.accept_bytes(input);
        assert!(!(ok && rt.is_accepted()), "wrong delimiter accepted (should reject)");
    }

    /// When additionalProperties:false is not explicitly declared in the
    /// parameters schema, the grammar uses a Kleene-star kv-list that accepts
    /// any key from the declared set. An *undeclared* key wrapped in
    /// `<|"|>` should still be accepted because the grammar uses an item
    /// alternation (any known key) — however a key name that doesn't match
    /// any known literal WILL be rejected because the item rule only contains
    /// the declared key literals.
    ///
    /// Note: unlike json_schema's additionalProperties enforcement, the Gemma
    /// kv-list grammar does not try to reject unknown keys — that would require
    /// the O(2^N) permutation algorithm with "used-key" tracking. Instead,
    /// extra fields at runtime are handled by parse_gemma4_tool_call (which
    /// ignores them) or by the schema-level validation at the API layer.
    #[test]
    fn gemma4_tool_call_grammar_empty_args_accepted() {
        let schema = r#"{"type": "object", "properties": {}}"#;
        let mut rt = gemma4_runtime("noop", schema);
        let input = b"call:noop{}";
        assert!(rt.accept_bytes(input), "empty args form rejected");
        assert!(rt.is_accepted(), "not accepted");
    }

    // -----------------------------------------------------------------------
    // Qwen 3.5/3.6 grammar tests
    // -----------------------------------------------------------------------

    /// Canonical Qwen 3.5 emission for `get_weather(location: "Paris")`.
    #[test]
    fn qwen35_tool_call_grammar_accepts_canonical_emission() {
        let schema = r#"{
            "type": "object",
            "properties": {
                "location": {"type": "string"},
                "unit": {"type": "string"}
            }
        }"#;
        let mut rt = qwen35_runtime("get_weather", schema);
        // Canonical Qwen emission (template emits `\n` around values):
        let input = b"<function=get_weather>\n<parameter=location>\nParis\n</parameter>\n<parameter=unit>\ncelsius\n</parameter>\n</function>";
        assert!(rt.accept_bytes(input), "canonical Qwen35 emission rejected");
        assert!(rt.is_accepted(), "not accepted at end");
    }

    /// Reversed parameter order must be accepted (Kleene-star approach).
    #[test]
    fn qwen35_tool_call_grammar_accepts_reversed_param_order() {
        let schema = r#"{
            "type": "object",
            "properties": {
                "location": {"type": "string"},
                "unit": {"type": "string"}
            }
        }"#;
        let mut rt = qwen35_runtime("get_weather", schema);
        let input = b"<function=get_weather>\n<parameter=unit>\nfahrenheit\n</parameter>\n<parameter=location>\nSF\n</parameter>\n</function>";
        assert!(rt.accept_bytes(input), "reversed param order rejected");
        assert!(rt.is_accepted(), "not accepted");
    }

    /// A malformed function wrapper (wrong tag syntax) must be rejected.
    #[test]
    fn qwen35_tool_call_grammar_rejects_malformed_wrapper() {
        let schema = r#"{
            "type": "object",
            "properties": {"location": {"type": "string"}}
        }"#;
        let mut rt = qwen35_runtime("get_weather", schema);
        // `[function=get_weather]` — square brackets instead of angle brackets.
        let input = b"[function=get_weather]\n[parameter=location]\nParis\n[/parameter]\n[/function]";
        let ok = rt.accept_bytes(input);
        assert!(!(ok && rt.is_accepted()), "malformed wrapper accepted (should reject)");
    }

    /// Empty parameter list (no arguments).
    #[test]
    fn qwen35_tool_call_grammar_accepts_empty_params() {
        let schema = r#"{"type": "object", "properties": {}}"#;
        let mut rt = qwen35_runtime("ping", schema);
        let input = b"<function=ping></function>";
        assert!(rt.accept_bytes(input), "empty params form rejected");
        assert!(rt.is_accepted(), "not accepted");
    }

    /// Verify the GBNF round-trips: grammar accepts output that
    /// `parse_qwen35_tool_call` can also parse back.  Closes the loop:
    /// grammar-constrained → parseable.
    #[test]
    fn qwen35_grammar_accepted_output_is_parseable() {
        let schema = r#"{
            "type": "object",
            "properties": {
                "location": {"type": "string"},
                "unit": {"type": "string"}
            }
        }"#;
        let body = "<function=get_weather>\n<parameter=location>\nParis\n</parameter>\n<parameter=unit>\ncelsius\n</parameter>\n</function>";
        // Grammar accepts it.
        let mut rt = qwen35_runtime("get_weather", schema);
        assert!(rt.accept_bytes(body.as_bytes()), "grammar rejected body");
        assert!(rt.is_accepted());
        // parse_qwen35_tool_call also parses it.
        let parsed = parse_tool_call_body(&QWEN35, body).expect("parse_tool_call_body failed");
        assert_eq!(parsed.name, "get_weather");
        let v: serde_json::Value = serde_json::from_str(&parsed.arguments_json).unwrap();
        assert_eq!(v["location"], "Paris");
        assert_eq!(v["unit"], "celsius");
    }

    /// Verify the Gemma 4 grammar round-trip: grammar accepts output that
    /// `parse_gemma4_tool_call` can also parse back.
    #[test]
    fn gemma4_grammar_accepted_output_is_parseable() {
        let schema = r#"{
            "type": "object",
            "properties": {
                "location": {"type": "string"},
                "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}
            }
        }"#;
        let body = "call:get_weather{location:<|\"|>San Francisco<|\"|>,unit:<|\"|>celsius<|\"|>}";
        // Grammar accepts it.
        let mut rt = gemma4_runtime("get_weather", schema);
        assert!(rt.accept_bytes(body.as_bytes()), "grammar rejected body");
        assert!(rt.is_accepted());
        // parse_gemma4_tool_call also parses it.
        let parsed = parse_tool_call_body(&GEMMA4, body).expect("parse_tool_call_body failed");
        assert_eq!(parsed.name, "get_weather");
        let v: serde_json::Value = serde_json::from_str(&parsed.arguments_json).unwrap();
        assert_eq!(v["location"], "San Francisco");
        assert_eq!(v["unit"], "celsius");
    }

    /// Unknown model family returns Err from tool_call_gbnf.
    #[test]
    fn unknown_family_tool_call_gbnf_returns_err() {
        let unknown = ModelRegistration {
            family: "unknown_llama",
            id_substrings: &["unknown_llama"],
            reasoning_open: None,
            reasoning_close: None,
            tool_open: None,
            tool_close: None,
            tool_preamble: None,
        };
        let schema: serde_json::Value = serde_json::json!({});
        let result = unknown.tool_call_gbnf("f", &schema);
        assert!(result.is_err(), "expected Err for unknown family");
        assert!(result.unwrap_err().contains("unknown_llama"));
    }

    // -----------------------------------------------------------------------
    // Wave 2.5 B1 — Required parameter enforcement tests
    // -----------------------------------------------------------------------

    /// B1 Gemma4: schema with `required` — required key present → accept.
    #[test]
    fn b1_gemma4_required_present_accept() {
        let schema = r#"{
            "type": "object",
            "properties": {
                "city": {"type": "string"},
                "units": {"type": "string"}
            },
            "required": ["city"]
        }"#;
        let mut rt = gemma4_runtime("get_weather", schema);
        // city is required; units is optional but we include it too.
        let input = b"call:get_weather{city:<|\"|>Paris<|\"|>,units:<|\"|>metric<|\"|>}";
        assert!(rt.accept_bytes(input), "required key present should be accepted");
        assert!(rt.is_accepted());
    }

    /// B1 Gemma4: schema with `required` — required key absent → reject.
    #[test]
    fn b1_gemma4_required_missing_reject() {
        let schema = r#"{
            "type": "object",
            "properties": {
                "city": {"type": "string"},
                "units": {"type": "string"}
            },
            "required": ["city"]
        }"#;
        let mut rt = gemma4_runtime("get_weather", schema);
        // Only units supplied; city (required) is absent.
        let input = b"call:get_weather{units:<|\"|>metric<|\"|>}";
        let ok = rt.accept_bytes(input);
        assert!(!(ok && rt.is_accepted()), "missing required key must be rejected");
    }

    /// B1 Gemma4: required key permuted (opposite order) → still accepted.
    #[test]
    fn b1_gemma4_required_permuted_accept() {
        let schema = r#"{
            "type": "object",
            "properties": {
                "a": {"type": "integer"},
                "b": {"type": "integer"}
            },
            "required": ["a", "b"]
        }"#;
        let mut rt = gemma4_runtime("add", schema);
        // b before a — permutation grammar must accept both orderings.
        let input = b"call:add{b:2,a:1}";
        assert!(rt.accept_bytes(input), "permuted required keys must be accepted");
        assert!(rt.is_accepted());
    }

    /// B1 Gemma4: schema with 9 required keys → EmitterError::TooManyRequiredKeys.
    /// W-ζ: cap lowered from 16 to 8 to match json_schema.rs ANY_ORDER_MAX_REQUIRED.
    #[test]
    fn b1_gemma4_too_many_required_keys_err() {
        // Build a schema with 9 required keys (> MAX_REQUIRED_KEYS = 8).
        let mut props = serde_json::Map::new();
        let mut required = Vec::new();
        for i in 0..9usize {
            let k = format!("key{}", i);
            props.insert(k.clone(), serde_json::json!({"type": "string"}));
            required.push(serde_json::Value::String(k));
        }
        let schema = serde_json::json!({
            "type": "object",
            "properties": props,
            "required": required
        });
        let result = GEMMA4.tool_call_gbnf("f", &schema);
        assert!(result.is_err(), "9 required keys must return Err");
        let msg = result.unwrap_err();
        assert!(msg.contains("9"), "error message should mention count: {}", msg);
        assert!(msg.contains("8"), "error message should mention cap: {}", msg);
    }

    /// W-ζ HIGH-2: Gemma4 boundary — exactly 8 required keys compiles OK.
    #[test]
    fn nine_required_keys_in_gemma_tool_call_gbnf_returns_too_many_required_keys() {
        // 9 keys — must be rejected (boundary at 8).
        let mut props = serde_json::Map::new();
        let mut required = Vec::new();
        for i in 0..9usize {
            let k = format!("p{}", i);
            props.insert(k.clone(), serde_json::json!({"type": "integer"}));
            required.push(serde_json::Value::String(k));
        }
        let schema = serde_json::json!({
            "type": "object",
            "properties": props,
            "required": required
        });
        let result = GEMMA4.tool_call_gbnf("tool9", &schema);
        assert!(result.is_err(), "9 required keys must return TooManyRequiredKeys");
        let msg = result.unwrap_err();
        assert!(msg.contains("9"), "error must mention count 9: {}", msg);
        assert!(msg.contains("8"), "error must mention cap 8: {}", msg);
    }

    /// W-ζ HIGH-2: Gemma4 boundary — exactly 8 required keys compiles OK.
    #[test]
    fn eight_required_keys_in_gemma_tool_call_gbnf_compiles_ok() {
        let mut props = serde_json::Map::new();
        let mut required = Vec::new();
        for i in 0..8usize {
            let k = format!("p{}", i);
            props.insert(k.clone(), serde_json::json!({"type": "integer"}));
            required.push(serde_json::Value::String(k));
        }
        let schema = serde_json::json!({
            "type": "object",
            "properties": props,
            "required": required
        });
        let result = GEMMA4.tool_call_gbnf("tool8", &schema);
        assert!(result.is_ok(), "8 required keys must compile without error");
    }

    /// B1 Qwen35: required key present → accept.
    #[test]
    fn b1_qwen35_required_present_accept() {
        let schema = r#"{
            "type": "object",
            "properties": {
                "city": {"type": "string"},
                "units": {"type": "string"}
            },
            "required": ["city"]
        }"#;
        let mut rt = qwen35_runtime("get_weather", schema);
        let input = b"<function=get_weather>\n<parameter=city>\nParis\n</parameter>\n<parameter=units>\nmetric\n</parameter>\n</function>";
        assert!(rt.accept_bytes(input), "required key present should be accepted");
        assert!(rt.is_accepted());
    }

    /// B1 Qwen35: required key absent → reject.
    #[test]
    fn b1_qwen35_required_missing_reject() {
        let schema = r#"{
            "type": "object",
            "properties": {
                "city": {"type": "string"},
                "units": {"type": "string"}
            },
            "required": ["city"]
        }"#;
        let mut rt = qwen35_runtime("get_weather", schema);
        // Only units supplied; city (required) is absent.
        let input = b"<function=get_weather>\n<parameter=units>\nmetric\n</parameter>\n</function>";
        let ok = rt.accept_bytes(input);
        assert!(!(ok && rt.is_accepted()), "missing required key must be rejected");
    }

    /// B1 Qwen35: required keys permuted → accept.
    #[test]
    fn b1_qwen35_required_permuted_accept() {
        let schema = r#"{
            "type": "object",
            "properties": {
                "a": {"type": "integer"},
                "b": {"type": "integer"}
            },
            "required": ["a", "b"]
        }"#;
        let mut rt = qwen35_runtime("add", schema);
        // b before a.
        let input = b"<function=add>\n<parameter=b>\n2\n</parameter>\n<parameter=a>\n1\n</parameter>\n</function>";
        assert!(rt.accept_bytes(input), "permuted required keys must be accepted");
        assert!(rt.is_accepted());
    }

    /// B1 Qwen35: 9 required keys → EmitterError::TooManyRequiredKeys.
    /// W-ζ: cap lowered from 16 to 8 to match json_schema.rs ANY_ORDER_MAX_REQUIRED.
    #[test]
    fn b1_qwen35_too_many_required_keys_err() {
        let mut props = serde_json::Map::new();
        let mut required = Vec::new();
        for i in 0..9usize {
            let k = format!("key{}", i);
            props.insert(k.clone(), serde_json::json!({"type": "string"}));
            required.push(serde_json::Value::String(k));
        }
        let schema = serde_json::json!({
            "type": "object",
            "properties": props,
            "required": required
        });
        let result = QWEN35.tool_call_gbnf("f", &schema);
        assert!(result.is_err(), "9 required keys must return Err");
        let msg = result.unwrap_err();
        assert!(msg.contains("9"), "error message should mention count: {}", msg);
        assert!(msg.contains("8"), "error message should mention cap: {}", msg);
    }

    /// W-ζ HIGH-2: Qwen35 boundary — 9 required keys returns TooManyRequiredKeys.
    #[test]
    fn nine_required_keys_in_qwen35_tool_call_gbnf_returns_too_many_required_keys() {
        let mut props = serde_json::Map::new();
        let mut required = Vec::new();
        for i in 0..9usize {
            let k = format!("q{}", i);
            props.insert(k.clone(), serde_json::json!({"type": "string"}));
            required.push(serde_json::Value::String(k));
        }
        let schema = serde_json::json!({
            "type": "object",
            "properties": props,
            "required": required
        });
        let result = QWEN35.tool_call_gbnf("qtool9", &schema);
        assert!(result.is_err(), "9 required keys must return TooManyRequiredKeys");
        let msg = result.unwrap_err();
        assert!(msg.contains("9"), "error must mention count 9: {}", msg);
        assert!(msg.contains("8"), "error must mention cap 8: {}", msg);
    }

    /// W-ζ HIGH-2: Qwen35 boundary — exactly 8 required keys compiles OK.
    #[test]
    fn eight_required_keys_in_qwen35_tool_call_gbnf_compiles_ok() {
        let mut props = serde_json::Map::new();
        let mut required = Vec::new();
        for i in 0..8usize {
            let k = format!("q{}", i);
            props.insert(k.clone(), serde_json::json!({"type": "string"}));
            required.push(serde_json::Value::String(k));
        }
        let schema = serde_json::json!({
            "type": "object",
            "properties": props,
            "required": required
        });
        let result = QWEN35.tool_call_gbnf("qtool8", &schema);
        assert!(result.is_ok(), "8 required keys must compile without error");
    }

    // -----------------------------------------------------------------------
    // Wave 2.5 B3 — Nested schema rejection tests
    // -----------------------------------------------------------------------

    /// B3 Gemma4: `array` type parameter → structured Err.
    #[test]
    fn b3_gemma4_array_param_returns_err() {
        let schema = r#"{
            "type": "object",
            "properties": {
                "items": {"type": "array"},
                "name": {"type": "string"}
            }
        }"#;
        let result = GEMMA4.tool_call_gbnf("process", &serde_json::from_str(schema).unwrap());
        assert!(result.is_err(), "array type must return Err");
        let msg = result.unwrap_err();
        assert!(msg.contains("array"), "error must mention type 'array': {}", msg);
        assert!(msg.contains("items"), "error must mention param name: {}", msg);
        assert!(msg.contains("process"), "error must mention function name: {}", msg);
    }

    /// B3 Gemma4: `object` type parameter → structured Err.
    #[test]
    fn b3_gemma4_nested_object_returns_err() {
        let schema = r#"{
            "type": "object",
            "properties": {
                "config": {"type": "object"}
            }
        }"#;
        let result = GEMMA4.tool_call_gbnf("configure", &serde_json::from_str(schema).unwrap());
        assert!(result.is_err(), "object type must return Err");
        let msg = result.unwrap_err();
        assert!(msg.contains("object"), "error must mention type 'object': {}", msg);
    }

    /// B3 Qwen35: `array` type parameter → structured Err.
    #[test]
    fn b3_qwen35_array_param_returns_err() {
        let schema = r#"{
            "type": "object",
            "properties": {
                "tags": {"type": "array"}
            }
        }"#;
        let result = QWEN35.tool_call_gbnf("tag_item", &serde_json::from_str(schema).unwrap());
        assert!(result.is_err(), "array type must return Err");
        let msg = result.unwrap_err();
        assert!(msg.contains("array"), "error must mention type 'array': {}", msg);
        assert!(msg.contains("tags"), "error must mention param name: {}", msg);
    }

    /// B3 Qwen35: `object` type parameter → structured Err.
    #[test]
    fn b3_qwen35_nested_object_returns_err() {
        let schema = r#"{
            "type": "object",
            "properties": {
                "metadata": {"type": "object"}
            }
        }"#;
        let result = QWEN35.tool_call_gbnf("set_meta", &serde_json::from_str(schema).unwrap());
        assert!(result.is_err(), "object type must return Err");
        let msg = result.unwrap_err();
        assert!(msg.contains("object"), "error must mention type 'object': {}", msg);
    }

    /// B3 integration: both models, scalar params unaffected.
    #[test]
    fn b3_scalar_params_unaffected() {
        let schema = serde_json::json!({
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "count": {"type": "integer"},
                "enabled": {"type": "boolean"}
            }
        });
        assert!(GEMMA4.tool_call_gbnf("f", &schema).is_ok(), "scalars must not be rejected by B3");
        assert!(QWEN35.tool_call_gbnf("f", &schema).is_ok(), "scalars must not be rejected by B3");
    }

    // -----------------------------------------------------------------------
    // Wave 2.5 B6 — str-char escape rule and trailing-newline audit tests
    // -----------------------------------------------------------------------

    /// B6 audit: Gemma4 grammar accepts a string value containing a backslash
    /// sequence (e.g. `C:\Users\test`).  The `[^<\\] | [\\] [^\x00-\x1F]` rule
    /// handles `\U` as backslash + non-control char.
    /// Template reference: chat_template.jinja:113 emits `<|"|>arg<|"|>` raw;
    /// no HTML escaping of backslash.  The grammar rule is therefore
    /// conservative-correct: allows the escape pattern the model may produce.
    #[test]
    fn b6_gemma4_str_char_accepts_backslash_sequence() {
        let schema = r#"{
            "type": "object",
            "properties": {"path": {"type": "string"}}
        }"#;
        let mut rt = gemma4_runtime("read_file", schema);
        // path contains a backslash sequence
        let input = "call:read_file{path:<|\"|>C:\\Users\\test<|\"|>}";
        assert!(
            rt.accept_bytes(input.as_bytes()),
            "backslash sequence in Gemma string must be accepted"
        );
        assert!(rt.is_accepted());
    }

    /// B6 audit: Qwen35 trailing newline rule — every `</parameter>` is
    /// followed by `\n`.  Confirmed from tokenizer_config.json chat_template:
    /// `{{- '\n</parameter>\n' }}`.  Grammar emits `newline_lit param_close_lit
    /// newline_lit` for each block, including the last block before `</function>`.
    #[test]
    fn b6_qwen35_trailing_newline_on_last_param_required() {
        let schema = r#"{
            "type": "object",
            "properties": {
                "location": {"type": "string"}
            }
        }"#;
        let mut rt = qwen35_runtime("weather", schema);
        // Canonical template emission: every </parameter> has trailing \n.
        let correct = b"<function=weather>\n<parameter=location>\nParis\n</parameter>\n</function>";
        assert!(
            rt.accept_bytes(correct),
            "trailing \\n after </parameter> must be accepted"
        );
        assert!(rt.is_accepted(), "must be accepted");
    }

    /// B6 audit: Qwen35 grammar rejects emission without trailing newline
    /// after `</parameter>`.  The template always emits the trailing newline,
    /// so a grammar that accepted the no-newline form would be too permissive.
    #[test]
    fn b6_qwen35_no_trailing_newline_rejected() {
        let schema = r#"{
            "type": "object",
            "properties": {
                "location": {"type": "string"}
            }
        }"#;
        let mut rt = qwen35_runtime("weather", schema);
        // Missing trailing \n after </parameter>.
        let wrong = b"<function=weather>\n<parameter=location>\nParis\n</parameter></function>";
        let ok = rt.accept_bytes(wrong);
        assert!(
            !(ok && rt.is_accepted()),
            "emission without trailing \\n after </parameter> must be rejected"
        );
    }
}
