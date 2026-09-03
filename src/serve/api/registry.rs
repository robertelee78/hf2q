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
        self.id_substrings
            .iter()
            .any(|s| lower.contains(&s.to_ascii_lowercase()))
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
    /// `shape` selects the root rule's overall structure (Wave 2.7 W-η):
    ///
    ///   * `GrammarShape::SingleBody` — root accepts ONLY the body
    ///     (e.g. Gemma `call:NAME{...}`, Qwen `<function=NAME>...</function>`).
    ///     Open/close markers are NOT in the grammar; the caller relies on
    ///     `ToolCallSplitter` to swallow them and on `awaiting_trigger`-gated
    ///     lazy enforcement (`GrammarKind::ToolCallBodyAuto`) so the runtime
    ///     only kicks in after the open marker is observed.  Pre-W-η
    ///     behaviour, preserved for forward-looking AUTO support.
    ///
    ///   * `GrammarShape::OneOrMoreCalls { parallel }` — root accepts the
    ///     FULL CALL: `marker_open + body + marker_close` (Gemma 4),
    ///     `\n<tool_call>\n<function=NAME>\n...\n</function>\n</tool_call>`
    ///     (Qwen 3.5/3.6), with `min_calls = 1` so at least one call is
    ///     produced.  When `parallel = true`, the root accepts repetition
    ///     (Gemma: bare `(call)+`; Qwen: `call ("\n" call)*`).  Used by
    ///     `compile_tool_grammar` for `tool_choice = required / function`
    ///     paired with `GrammarKind::ToolCallBodyRequired` for eager
    ///     enforcement from token 0.  Mirrors the peer's
    ///     `p.repeat(call, min=1, max=parallel?-1:1)`.
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
        shape: GrammarShape,
    ) -> Result<String, String> {
        let variants =
            crate::serve::api::grammar::json_schema::normalize_schema_variants(params_schema)
                .map_err(|error| error.to_string())?;
        let mut grammars = Vec::with_capacity(variants.len());
        for variant in variants {
            validate_native_tool_schema_compat(&variant, "")?;
            let grammar = match self.family {
                "gemma4" => gemma4_tool_call_gbnf(fn_name, &variant, shape)
                    .map_err(|error| error.to_string())?,
                "qwen35" => qwen35_tool_call_gbnf(fn_name, &variant, shape)
                    .map_err(|error| error.to_string())?,
                "deepseek4" => deepseek4_tool_call_gbnf(fn_name, &variant, shape)
                    .map_err(|error| error.to_string())?,
                other => {
                    return Err(format!(
                        "tool_call_gbnf: no per-model grammar emitter for family '{}'",
                        other
                    ))
                }
            };
            grammars.push(grammar);
        }
        combine_schema_variant_grammars(grammars)
    }

    /// Per-family separator that appears between calls 2+ in a parallel
    /// tool-call sequence.  Returned for the multi-function combiner case
    /// (`combine_function_grammars`) so the alternated-across-functions
    /// repetition uses the family-correct chat-template form.
    ///
    /// * Gemma 4: empty string — chat_template.jinja:189-205 emits calls
    ///   back-to-back with no separator (`<|tool_call>...<tool_call|>`
    ///   immediately followed by the next call's `<|tool_call>`).
    /// * Qwen 3.5/3.6: `"\n"` — tokenizer_config.json:285+ chat_template's
    ///   loop "else" branch prepends `\n` to calls 2+.
    /// * Unknown families: empty string (no separator) is the safe
    ///   default; the per-family emitter rejects unknown families before
    ///   this method is called from `combine_function_grammars`.
    pub fn parallel_call_separator(&self) -> &'static str {
        match self.family {
            "gemma4" => "",
            "qwen35" => "\n",
            "deepseek4" => "\n",
            _ => "",
        }
    }

    /// Wave 3.5 HIGH-1 — Auto-lazy multi-function parallel inter-call
    /// sequence.  Used by `combine_function_grammars` when wiring up
    /// `tool_choice = auto` with multiple registered functions and
    /// `parallel_tool_calls = true`.
    ///
    /// Unlike `parallel_call_separator()`, this string includes the
    /// **open marker** because the per-function alts under Auto-lazy
    /// are body-only (`OneOrMoreCallsBodyOnly { parallel: false }` =
    /// `body close`).  Inserting the open marker as part of the
    /// inter-call sequence reconstructs the correct chat-template
    /// emission shape WITHOUT requiring the FIRST open marker (which
    /// the splitter consumes via the awaiting_trigger gate before the
    /// grammar is engaged).
    ///
    /// * Gemma 4: `"<|tool_call>"` — no whitespace, just the open
    ///   marker (chat_template.jinja:189-205 emits calls back-to-back
    ///   with the next call's `<|tool_call>` immediately after the
    ///   previous `<tool_call|>`).
    /// * Qwen 3.5/3.6: `"\n<tool_call>\n"` — `\n` separator + open
    ///   marker + `\n` (mirrors the chat_template loop's
    ///   `\n<tool_call>\n...\n</tool_call>` per-call pattern; the
    ///   trailing `\n</tool_call>` is provided by the per-fn body-only
    ///   shape).
    /// * Unknown families: empty string (combiner rejects unknown
    ///   families before this method is called).
    pub fn auto_lazy_multi_fn_inter_call(&self) -> &'static str {
        match self.family {
            "gemma4" => "<|tool_call>",
            "qwen35" => "\n<tool_call>\n",
            // DeepSeek puts multiple invokes inside one outer block. The
            // dedicated emitter carries that repetition; multi-function
            // combination uses a newline between body alternatives.
            "deepseek4" => "\n",
            _ => "",
        }
    }
}

fn validate_native_tool_schema_compat(
    schema: &serde_json::Value,
    path: &str,
) -> Result<(), String> {
    let Some(object) = schema.as_object() else {
        return Ok(());
    };
    for keyword in ["minProperties", "maxProperties", "prefixItems"] {
        if object.contains_key(keyword) {
            return Err(format!(
                "native tool wire cannot enforce JSON Schema assertion at {}/{}; use a response structured-output constraint or remove the assertion",
                if path.is_empty() { "" } else { path },
                keyword
            ));
        }
    }
    for container in ["properties", "$defs", "definitions"] {
        if let Some(children) = object.get(container).and_then(serde_json::Value::as_object) {
            for (name, child) in children {
                validate_native_tool_schema_compat(child, &format!("{path}/{container}/{name}"))?;
            }
        }
    }
    for keyword in ["items", "additionalProperties"] {
        if let Some(child) = object.get(keyword) {
            validate_native_tool_schema_compat(child, &format!("{path}/{keyword}"))?;
        }
    }
    for keyword in ["anyOf", "oneOf", "allOf"] {
        if let Some(children) = object.get(keyword).and_then(serde_json::Value::as_array) {
            for (index, child) in children.iter().enumerate() {
                validate_native_tool_schema_compat(child, &format!("{path}/{keyword}/{index}"))?;
            }
        }
    }
    Ok(())
}

/// Root-rule shape selector for `tool_call_gbnf`.
///
/// Wave 2.7 W-η Q-A + Q-B: unified emitter API for the eager-grammar
/// (REQUIRED/Function) and forward-looking lazy-grammar (AUTO) paths.
///
/// See `ModelRegistration::tool_call_gbnf` for the per-variant semantics.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GrammarShape {
    /// Body-only root.  Open/close markers handled by the splitter; the
    /// runtime is expected to be `awaiting_trigger`-gated by the caller
    /// (`GrammarKind::ToolCallBodyAuto`).  Pre-Wave 2.7 behaviour.
    SingleBody,
    /// Full-call root (`marker_open + body + marker_close`, repeated
    /// `min=1`, `max=parallel?∞:1`).  Open/close markers ARE in the
    /// grammar; the runtime is expected to be EAGER from token 0
    /// (`GrammarKind::ToolCallBodyRequired`).
    ///
    /// `parallel = true` accepts repetition (Wave 2.7 W-η Q-B):
    ///   * Gemma 4: bare back-to-back `(call)+` (no separator —
    ///     chat_template.jinja:189-205 emits calls with no inter-call
    ///     bytes).
    ///   * Qwen 3.5/3.6: `call ("\n" call)*` (template separates calls
    ///     2+ with literal `\n`; tokenizer_config.json:285+).
    OneOrMoreCalls { parallel: bool },
    /// Wave 3.5 HIGH-1 — body-only multi-call root with the **first** open
    /// marker stripped.  Used exclusively by `tool_choice = auto` paired
    /// with `GrammarKind::ToolCallBodyAuto` (lazy / awaiting_trigger).
    ///
    /// Production-order rationale (audit
    /// /tmp/cfa-cfa-20260427-adr005-wave3/codex-review-last.txt divergence
    /// "W-B2 Auto lazy grammar production correctness"):  the engine's
    /// decode loop calls `runtime.accept_bytes(token_bytes)` on every
    /// sampled token BEFORE the splitter's marker boundary fires.  While
    /// `awaiting_trigger=true` (Auto-lazy), `accept_bytes` is a no-op.
    /// The splitter then sees the per-model open marker as a special
    /// single-token emission (see the marker-shape contract on
    /// `tool_open` / `tool_close` above), fires `ToolCallOpen`, and the
    /// engine calls `runtime.trigger()` flipping the gate false.  The
    /// FIRST open marker has already been consumed (as no-op) at this
    /// point — replaying it into the now-eager grammar would distort
    /// the byte stream.
    ///
    /// Required/Function uses `OneOrMoreCalls` because no splitter
    /// precedes the eager grammar; the marker IS part of the body
    /// grammar from byte 0.  Auto uses this variant because the FIRST
    /// open marker is consumed by the awaiting_trigger gate before the
    /// grammar is engaged.  Subsequent markers (close marker after this
    /// call's body, AND any inter-call open markers under
    /// `parallel = true`) ARE in the grammar — those bytes flow through
    /// `accept_bytes` while the runtime is eager.
    ///
    /// Concrete shape:
    ///   * `parallel = false`:
    ///       Gemma 4: `body close_marker space`
    ///       Qwen 3.5/3.6: `body \n close_marker space` (mirrors the
    ///         chat_template's `\n</tool_call>` close form)
    ///   * `parallel = true`:
    ///       Gemma 4: `body close_marker ( open_marker body close_marker )* space`
    ///       Qwen 3.5/3.6: `body \n close_marker ( \n open_marker \n body \n close_marker )* space`
    ///
    /// Audit recommendation reference (research-report.md §Q1+§Q2): the
    /// canonical Wave 2.6 architecture is "grammar = body validator;
    /// splitter = boundary detector".  For Auto-lazy the FIRST marker is
    /// the splitter's responsibility (it's what triggers the grammar);
    /// thereafter the grammar accepts everything including any closing
    /// and re-opening markers.
    OneOrMoreCallsBodyOnly { parallel: bool },
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

/// Qwen 3.5-family models, including Qwen3.6 and Qwen3.8 releases. Uses
/// `<think>` / `</think>` — the Qwen convention
/// (no pipe in the closer; distinct from Gemma's). Tool calling also uses
/// `<tool_call>` / `</tool_call>` (Qwen standard).
pub const QWEN35: ModelRegistration = ModelRegistration {
    family: "qwen35",
    id_substrings: &[
        "qwen3.5", "qwen3.6", "qwen3.8", "qwen35", "qwen36", "qwen38",
    ],
    reasoning_open: Some("<think>"),
    reasoning_close: Some("</think>"),
    tool_open: Some("<tool_call>"),
    tool_close: Some("</tool_call>"),
    tool_preamble: None,
};

/// DeepSeek-V4-Flash-0731. Tool calls use one outer DSML block containing
/// one or more invoke elements; the body parser therefore returns a vector
/// rather than assuming one marker pair equals one function call.
pub const DEEPSEEK4: ModelRegistration = ModelRegistration {
    family: "deepseek4",
    id_substrings: &["deepseek-v4", "deepseek v4", "deepseek4", "deepseek_v4"],
    reasoning_open: Some("<think>"),
    reasoning_close: Some("</think>"),
    tool_open: Some("<｜DSML｜tool_calls>"),
    tool_close: Some("</｜DSML｜tool_calls>"),
    tool_preamble: None,
};

/// All built-in registrations in priority order. Later entries override
/// earlier ones when substrings overlap — but day-one substrings are
/// disjoint.
pub const BUILTIN_REGISTRATIONS: &[ModelRegistration] = &[GEMMA4, QWEN35, DEEPSEEK4];

// ---------------------------------------------------------------------------
// Registry (process-global)
// ---------------------------------------------------------------------------

/// Dynamic registry, seeded with `BUILTIN_REGISTRATIONS` plus any runtime
/// additions via `register`.
static REGISTRY: OnceLock<std::sync::RwLock<Vec<ModelRegistration>>> = OnceLock::new();

fn reg() -> &'static std::sync::RwLock<Vec<ModelRegistration>> {
    REGISTRY.get_or_init(|| std::sync::RwLock::new(BUILTIN_REGISTRATIONS.to_vec()))
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
    /// Qwen may transition directly from thinking into a native tool call
    /// without first emitting `</think>`. Treat the tool opener as an
    /// implicit reasoning boundary, but preserve it for ToolCallSplitter.
    implicit_reasoning_close_marker: Option<&'static str>,
    /// Some native templates place a structural separator immediately after
    /// the reasoning close marker. That separator belongs to the wire
    /// protocol, not to OpenAI `content`.
    post_close_separator: &'static str,
    awaiting_post_close_separator: bool,
    in_reasoning: bool,
    /// Sliding tail of decoded text — long enough to see either marker span
    /// across token boundaries.
    tail_buf: String,
    tail_cap: usize,
}

impl ReasoningSplitter {
    /// Build from a registration. If the registration has no reasoning
    /// markers, returns `None` — callers route all text to `Content`.
    ///
    /// Starts OUTSIDE reasoning. Engine paths must construct through
    /// [`make_reasoning_splitter`] instead so the ADR-005 iter-230
    /// forced-open seed is never dropped (grep-pinned by
    /// `iter230_b2_factory_only_construction`).
    pub fn from_registration(reg: &ModelRegistration) -> Option<Self> {
        Self::from_registration_forced(reg, false)
    }

    /// Build from a registration with an explicit initial reasoning
    /// state (ADR-005 iter-230 B). `forced_open = true` when the
    /// rendered prompt ends inside an open reasoning block (e.g. the
    /// Qwen 3.6 template seeds `<think>\n` at the end of the generation
    /// prompt with thinking on) — the completion then begins INSIDE
    /// reasoning and the model never re-emits the open marker, so an
    /// unseeded splitter would leak the whole reasoning span plus the
    /// close marker into `content`. A redundant model-emitted open
    /// marker while seeded open is reasoning TEXT (today's nested-open
    /// behavior), not swallowed.
    pub fn from_registration_forced(reg: &ModelRegistration, forced_open: bool) -> Option<Self> {
        let (open, close) = match (reg.reasoning_open, reg.reasoning_close) {
            (Some(o), Some(c)) if !o.is_empty() && !c.is_empty() => (o, c),
            _ => return None,
        };
        let implicit_reasoning_close_marker =
            (reg.family == "qwen35").then_some(reg.tool_open).flatten();
        let cap = open
            .len()
            .max(close.len())
            .max(implicit_reasoning_close_marker.map_or(0, str::len))
            .max(1);
        Some(Self {
            open_marker: open,
            close_marker: close,
            implicit_reasoning_close_marker,
            post_close_separator: if close == "</think>" { "\n\n" } else { "" },
            awaiting_post_close_separator: false,
            in_reasoning: forced_open,
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
            if self.awaiting_post_close_separator {
                let remainder = &scan[scan_cursor..];
                if remainder.len() < self.post_close_separator.len()
                    && self.post_close_separator.starts_with(remainder)
                {
                    self.tail_buf = scan[out_cursor..].to_string();
                    break;
                }
                if remainder.starts_with(self.post_close_separator) {
                    scan_cursor += self.post_close_separator.len();
                    out_cursor = scan_cursor;
                }
                self.awaiting_post_close_separator = false;
            }
            let (active_marker, implicit_close) = if self.in_reasoning {
                let close = scan[scan_cursor..]
                    .find(self.close_marker)
                    .map(|offset| (offset, self.close_marker, false));
                let implicit = self.implicit_reasoning_close_marker.and_then(|marker| {
                    scan[scan_cursor..]
                        .find(marker)
                        .map(|offset| (offset, marker, true))
                });
                match (close, implicit) {
                    (Some(close), Some(implicit)) if implicit.0 < close.0 => {
                        (implicit.1, implicit.2)
                    }
                    (Some(close), _) => (close.1, close.2),
                    (None, Some(implicit)) => (implicit.1, implicit.2),
                    (None, None) => (self.close_marker, false),
                }
            } else {
                (self.open_marker, false)
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
                    // A Qwen tool opener closes the reasoning channel
                    // implicitly, but it is input to the downstream tool
                    // splitter and therefore must not be swallowed here.
                    if implicit_close {
                        self.in_reasoning = false;
                        scan_cursor = marker_start;
                        out_cursor = marker_start;
                        continue;
                    }
                    // Flip state + skip past marker.
                    let closed_reasoning = self.in_reasoning;
                    self.in_reasoning = !self.in_reasoning;
                    scan_cursor = marker_start + active_marker.len();
                    out_cursor = scan_cursor;
                    self.awaiting_post_close_separator =
                        closed_reasoning && !self.post_close_separator.is_empty();
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
        if self.awaiting_post_close_separator
            && self.post_close_separator.starts_with(&self.tail_buf)
        {
            self.tail_buf.clear();
            self.awaiting_post_close_separator = false;
        }
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
/// Per-family in-call resync markers. When `ToolCallSplitter::feed` sees
/// any of these literals while `in_tool_call=true`, the current call body
/// is aborted (synthetic `ToolCallClose` emitted with the partial body)
/// rather than absorbing the marker bytes into `ToolCallText`. iter-219c
/// 2026-05-01 — closes the iter-218 LIVE-bug class structurally at the
/// splitter level instead of relying on downstream defensive scrubbing.
///
/// Marker selection mirrors the registered family special-tokens that
/// MUST NOT appear inside a tool-call body per the chat-template
/// `tokenizer_config.json`'s regex: for Gemma 4, the call body is
/// `<\|tool_call>(.*?)<tool_call\|>` — anything else (channel / turn /
/// tool_response) is OUT-of-call. Seeing one of those mid-call indicates
/// the model went off-template and the call MUST be aborted defensively.
fn family_resync_markers(open_marker: &str) -> &'static [&'static str] {
    match open_marker {
        "<|tool_call>" => &[
            // Gemma 4 family.
            "<|tool_response>",
            "<tool_response|>",
            "<|channel>",
            "<channel|>",
            "<|turn>",
            "<turn|>",
        ],
        "<tool_call>" => &[
            // Qwen 3.5/3.6 family.
            "<think>", "</think>",
        ],
        _ => &[],
    }
}

#[derive(Debug, Clone)]
pub struct ToolCallSplitter {
    open_marker: &'static str,
    close_marker: &'static str,
    /// iter-219c (2026-05-01): markers that, when observed inside a
    /// tool-call body (`in_tool_call=true`), trigger a synthetic
    /// `ToolCallClose` (abort). Derived from `family_resync_markers` at
    /// construction.
    in_call_resync: &'static [&'static str],
    in_tool_call: bool,
    /// Sliding tail of decoded text — long enough to see either marker span
    /// across token boundaries. See `ReasoningSplitter::tail_buf` for the
    /// identical mechanism. iter-219c: tail_cap now must accommodate the
    /// LONGEST resync marker too.
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
        let in_call_resync = family_resync_markers(open);
        // tail_cap must accommodate the longest of: open, close, any resync
        // marker — so all marker types are detectable across token boundaries.
        let mut cap = open.len().max(close.len()).max(1);
        for m in in_call_resync {
            cap = cap.max(m.len());
        }
        Some(Self {
            open_marker: open,
            close_marker: close,
            in_call_resync,
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
            // iter-219c (2026-05-01): in BOTH states, scan for the
            // active state-flip marker AND any stray family markers
            // that should be swallowed.
            //
            //   in_tool_call=true:
            //     - close_marker → state-flip (ToolCallClose)
            //     - resync markers (`<|tool_response>`, `<|channel>`, `<|turn>`)
            //       → state-flip (synthetic ToolCallClose, call aborted) +
            //       swallow the marker bytes
            //   in_tool_call=false:
            //     - open_marker → state-flip (ToolCallOpen)
            //     - close_marker (stray) → swallow (no event; just advance)
            //     - resync markers (stray) → swallow (no event; just advance)
            //
            // The "swallow stray" path keeps registered family markers
            // out of `delta.content` even when they appear in unexpected
            // positions (e.g. post-abort tail bytes after iter-219c
            // synthesizes a ToolCallClose). Without this, a `<tool_call|>`
            // following an aborted call would bleed into Content.
            #[derive(Clone, Copy)]
            enum Hit {
                StateFlip(&'static str),
                Swallow(&'static str),
            }
            let hit: Option<(usize, Hit)> = if self.in_tool_call {
                let mut best: Option<(usize, Hit)> = scan[scan_cursor..]
                    .find(self.close_marker)
                    .map(|r| (r, Hit::StateFlip(self.close_marker)));
                for &resync in self.in_call_resync {
                    if let Some(r) = scan[scan_cursor..].find(resync) {
                        match best {
                            None => best = Some((r, Hit::StateFlip(resync))),
                            Some((b, _)) if r < b => best = Some((r, Hit::StateFlip(resync))),
                            _ => {}
                        }
                    }
                }
                best
            } else {
                let mut best: Option<(usize, Hit)> = scan[scan_cursor..]
                    .find(self.open_marker)
                    .map(|r| (r, Hit::StateFlip(self.open_marker)));
                // Stray close_marker outside a call → swallow.
                if let Some(r) = scan[scan_cursor..].find(self.close_marker) {
                    match best {
                        None => best = Some((r, Hit::Swallow(self.close_marker))),
                        Some((b, _)) if r < b => best = Some((r, Hit::Swallow(self.close_marker))),
                        _ => {}
                    }
                }
                // Stray in-call markers outside a call → swallow.
                for &resync in self.in_call_resync {
                    if let Some(r) = scan[scan_cursor..].find(resync) {
                        match best {
                            None => best = Some((r, Hit::Swallow(resync))),
                            Some((b, _)) if r < b => best = Some((r, Hit::Swallow(resync))),
                            _ => {}
                        }
                    }
                }
                best
            };
            match hit {
                Some((rel, kind)) => {
                    let marker_start = scan_cursor + rel;
                    let marker = match kind {
                        Hit::StateFlip(m) | Hit::Swallow(m) => m,
                    };
                    // Emit text [out_cursor..marker_start] as the current slot.
                    if marker_start > out_cursor {
                        let text = scan[out_cursor..marker_start].to_string();
                        if self.in_tool_call {
                            out.push(ToolCallEvent::ToolCallText(text));
                        } else {
                            out.push(ToolCallEvent::Content(text));
                        }
                    }
                    match kind {
                        Hit::StateFlip(_) => {
                            // Emit the open/close synthetic event + flip state.
                            if self.in_tool_call {
                                out.push(ToolCallEvent::ToolCallClose);
                            } else {
                                out.push(ToolCallEvent::ToolCallOpen);
                            }
                            self.in_tool_call = !self.in_tool_call;
                        }
                        Hit::Swallow(_) => {
                            // Marker swallowed — no event, no state change.
                        }
                    }
                    scan_cursor = marker_start + marker.len();
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

/// How Qwen's native top-level parameter wire should be interpreted.
///
/// Qwen emits declared strings as raw text but serializes non-strings as
/// compact JSON.  Bytes such as `true`, `42`, `{}`, or `[]` are therefore
/// ambiguous without the request schema; this request-owned map preserves the
/// declared OpenAI tool type through both unary and incremental SSE parsing.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum ToolArgumentWireKind {
    RawString,
    Json,
    /// Untyped or mixed string/non-string schema.  Preserve the historical
    /// JSON-first/string-fallback interpretation because the native wire is
    /// not injective for this shape.
    Infer,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ToolArgumentWireKinds {
    functions: std::collections::BTreeMap<
        String,
        std::collections::BTreeMap<String, ToolArgumentWireKind>,
    >,
}

impl ToolArgumentWireKinds {
    pub fn kind_for(&self, function: &str, parameter: &str) -> Option<ToolArgumentWireKind> {
        self.functions
            .get(function)
            .and_then(|parameters| parameters.get(parameter))
            .copied()
    }

    pub fn function_count(&self) -> usize {
        self.functions.len()
    }
}

fn combine_wire_kinds<I>(kinds: I) -> ToolArgumentWireKind
where
    I: IntoIterator<Item = ToolArgumentWireKind>,
{
    let mut kinds = kinds.into_iter();
    let Some(first) = kinds.next() else {
        return ToolArgumentWireKind::Infer;
    };
    if kinds.all(|kind| kind == first) {
        first
    } else {
        ToolArgumentWireKind::Infer
    }
}

/// Classify the top-level value surface produced by Qwen's chat template.
/// Names never participate: only the JSON Schema controls interpretation.
pub fn qwen35_top_level_wire_kind(schema: &serde_json::Value) -> ToolArgumentWireKind {
    let Some(object) = schema.as_object() else {
        return ToolArgumentWireKind::Infer;
    };

    let mut constraints = Vec::new();
    if let Some(schema_type) = object.get("type") {
        let type_kind = |name: &str| match name {
            "string" => ToolArgumentWireKind::RawString,
            "integer" | "number" | "boolean" | "null" | "object" | "array" => {
                ToolArgumentWireKind::Json
            }
            _ => ToolArgumentWireKind::Infer,
        };
        match schema_type {
            serde_json::Value::String(name) => constraints.push(type_kind(name)),
            serde_json::Value::Array(names) => constraints.push(combine_wire_kinds(
                names
                    .iter()
                    .filter_map(serde_json::Value::as_str)
                    .map(type_kind),
            )),
            _ => constraints.push(ToolArgumentWireKind::Infer),
        }
    }
    if let Some(values) = object.get("enum").and_then(serde_json::Value::as_array) {
        constraints.push(combine_wire_kinds(values.iter().map(|value| {
            if value.is_string() {
                ToolArgumentWireKind::RawString
            } else {
                ToolArgumentWireKind::Json
            }
        })));
    }
    if let Some(value) = object.get("const") {
        constraints.push(if value.is_string() {
            ToolArgumentWireKind::RawString
        } else {
            ToolArgumentWireKind::Json
        });
    }
    for union in ["anyOf", "oneOf"] {
        if let Some(branches) = object.get(union).and_then(serde_json::Value::as_array) {
            constraints.push(combine_wire_kinds(
                branches.iter().map(qwen35_top_level_wire_kind),
            ));
        }
    }
    combine_wire_kinds(constraints)
}

/// Build the authoritative Qwen parameter map for one OpenAI tools request.
/// Duplicate function declarations are request errors. Even schemas that
/// collapse to the same wire kind can carry different enum/object/array
/// constraints, so accepting a duplicate would make executor identity
/// ambiguous.
pub fn qwen35_tool_argument_wire_kinds(
    tools: &[super::schema::Tool],
) -> Result<ToolArgumentWireKinds, String> {
    let mut functions = std::collections::BTreeMap::new();
    for tool in tools {
        let mut parameters = std::collections::BTreeMap::new();
        if let Some(properties) = tool
            .function
            .parameters
            .as_ref()
            .and_then(serde_json::Value::as_object)
            .and_then(|object| object.get("properties"))
            .and_then(serde_json::Value::as_object)
        {
            for (name, schema) in properties {
                parameters.insert(name.clone(), qwen35_top_level_wire_kind(schema));
            }
        }
        match functions.entry(tool.function.name.clone()) {
            std::collections::btree_map::Entry::Occupied(_) => {
                return Err(format!(
                    "duplicate tool '{}' is ambiguous",
                    tool.function.name
                ));
            }
            std::collections::btree_map::Entry::Vacant(entry) => {
                entry.insert(parameters);
            }
        }
    }
    Ok(ToolArgumentWireKinds { functions })
}

/// Decode one Qwen top-level parameter using the request's authoritative
/// schema contract.  `None` keeps legacy direct callers compatible; `Some`
/// fails closed for an unknown function/key instead of guessing a type.
pub fn decode_qwen35_parameter_value(
    function: &str,
    parameter: &str,
    raw: &str,
    wire_kinds: Option<&ToolArgumentWireKinds>,
) -> Option<serde_json::Value> {
    let kind = match wire_kinds {
        Some(kinds) => kinds.kind_for(function, parameter)?,
        None => ToolArgumentWireKind::Infer,
    };
    match kind {
        ToolArgumentWireKind::RawString => Some(serde_json::Value::String(raw.to_string())),
        ToolArgumentWireKind::Json => serde_json::from_str(raw).ok(),
        ToolArgumentWireKind::Infer => Some(
            serde_json::from_str(raw)
                .unwrap_or_else(|_| serde_json::Value::String(raw.to_string())),
        ),
    }
}

/// Every registered in-call special-token marker across all supported
/// families. Mirrors `BUILTIN_REGISTRATIONS` and the `LEAK_MARKERS` array
/// in `tests/openwebui_multiturn.rs:115-138`. Used by the Auto-policy
/// content-fallback scrubber so a malformed-body emit cannot leak any of
/// these tokens into `delta.content`.
const ALL_FAMILY_LEAK_MARKERS: &[&str] = &[
    // Gemma 4 reasoning + tool + turn
    "<|channel>",
    "<channel|>",
    "<|tool_call>",
    "<tool_call|>",
    "<|tool_response>",
    "<tool_response|>",
    "<|turn>",
    "<turn|>",
    // Qwen 3.5 / 3.6 reasoning + tool
    "<think>",
    "</think>",
    "<tool_call>",
    "</tool_call>",
    // DeepSeek-V4 DSML tool block + inner records
    "<｜DSML｜tool_calls>",
    "</｜DSML｜tool_calls>",
    "<｜DSML｜invoke",
    "</｜DSML｜invoke>",
    "<｜DSML｜parameter",
    "</｜DSML｜parameter>",
];

/// Scrub all registered family special-token markers from `body` so they
/// cannot leak into `delta.content` via the Auto-policy content-fallback
/// path (`emit_streaming_tool_call_close` on parse failure). Returns the
/// body with markers replaced by the empty string.
///
/// **iter-219b (2026-05-01)**: when the validity gate (`is_valid_tool_name`)
/// rejects a polluted tool-call name, `emit_streaming_tool_call_close`
/// under `ToolCallPolicy::Auto` falls back to emitting the raw body as
/// `Content`. Without scrubbing, the body's special-token literals
/// (e.g. `<|tool_response>` decoded from token id 50) reach the OpenAI
/// client verbatim — re-introducing the iter-217-class leak that the
/// `tests/openwebui_multiturn.rs::assert_no_leaked_special_tokens` gate is
/// supposed to make loud at LIVE-test time. Strip every registered
/// in-call marker so the fallback path stays content-correct regardless
/// of which family registered the request.
pub fn scrub_special_tokens(body: &str) -> String {
    let mut out = body.to_string();
    for m in ALL_FAMILY_LEAK_MARKERS {
        if out.contains(m) {
            out = out.replace(m, "");
        }
    }
    out
}

/// Validate an OpenAI-spec function name. Tool names are required to match
/// `^[a-zA-Z0-9_-]+$` per the OpenAI tool-spec — no special-token bytes,
/// no whitespace, no nested call/colon syntax.
///
/// **iter-219b (2026-05-01)**: this gate exists because the model can emit
/// known special-token literals (`<|tool_response>`, `<|channel>`, etc.)
/// MID-tool-call-body; the splitter — which only knows the tool-call
/// open/close marker pair — concatenates those literals into the body
/// buffer verbatim. Without name validation, `extract_gemma4_name_prefix`
/// (engine.rs) and `parse_gemma4_tool_call` / `parse_qwen35_tool_call`
/// below cheerfully return polluted names like
/// `get_current<|tool_response>call:get_current_weather` to the SSE
/// encoder, which proxies them straight to OpenAI clients as
/// `delta.tool_calls.function.name` — surfaced in iter-218 LIVE testing
/// as the malformed `get_currentcall:get_current_weather`. Returning
/// `None`/false from the validity check triggers the existing Auto-policy
/// content-fallback path, which is the OpenAI-spec-correct behavior for
/// an unparseable tool-call body.
pub fn is_valid_tool_name(name: &str) -> bool {
    !name.is_empty()
        && name
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-')
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
    parse_tool_call_body_with_wire_kinds(reg, body, None)
}

pub fn parse_tool_call_body_with_wire_kinds(
    reg: &ModelRegistration,
    body: &str,
    wire_kinds: Option<&ToolArgumentWireKinds>,
) -> Option<ParsedToolCall> {
    parse_tool_call_bodies_with_wire_kinds(reg, body, wire_kinds)?
        .into_iter()
        .next()
}

/// Parse all function calls represented by one registered outer tool block.
/// Gemma/Qwen marker pairs each contain one call; DeepSeek DSML deliberately
/// permits multiple invokes inside a single pair.
pub fn parse_tool_call_bodies(reg: &ModelRegistration, body: &str) -> Option<Vec<ParsedToolCall>> {
    parse_tool_call_bodies_with_wire_kinds(reg, body, None)
}

pub fn parse_tool_call_bodies_with_wire_kinds(
    reg: &ModelRegistration,
    body: &str,
    wire_kinds: Option<&ToolArgumentWireKinds>,
) -> Option<Vec<ParsedToolCall>> {
    match reg.family {
        "gemma4" => parse_gemma4_tool_call(body).map(|call| vec![call]),
        "qwen35" => parse_qwen35_tool_call_with_wire_kinds(body, wire_kinds).map(|call| vec![call]),
        "deepseek4" => crate::core::deepseek_v4_encoding::parse_tool_calls_body(body)
            .ok()
            .map(|calls| {
                calls
                    .into_iter()
                    .map(|call| ParsedToolCall {
                        name: call.function.name,
                        arguments_json: call.function.arguments,
                    })
                    .collect()
            }),
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
    // iter-219b (2026-05-01): reject names that contain special-token
    // bytes the splitter couldn't trim (e.g. `<|tool_response>` emitted
    // mid-call). OpenAI spec requires `[a-zA-Z0-9_-]+`. Returning None
    // triggers the existing content-fallback path.
    if !is_valid_tool_name(&name) {
        return None;
    }
    // Match braces — body of args ends at the LAST `}` (Gemma can have
    // nested args via the `<|"|>` quote-markers, but the outer brace is
    // the call boundary).
    let after_open = &rest[brace_start + 1..];
    let close_idx = after_open.rfind('}')?;
    let kv_str = &after_open[..close_idx];

    // iter-231b: split on top-level commas (depth-aware — commas inside
    // nested `{...}`/`[...]` values do NOT split), then decode each value
    // recursively so structured arguments survive as JSON objects/arrays
    // in `arguments_json` instead of being mangled into string fields.
    let kvs = split_gemma4_top_level(kv_str);
    let mut args = serde_json::Map::new();
    for kv in kvs {
        let (k, v) = split_gemma4_kv_once(kv)?;
        let key = k.trim().to_string();
        if key.is_empty() {
            return None;
        }
        args.insert(key, gemma4_value_to_json(v.trim()));
    }
    let arguments_json = serde_json::to_string(&serde_json::Value::Object(args)).ok()?;
    Some(ParsedToolCall {
        name,
        arguments_json,
    })
}

/// Split a Gemma 4 kv-list / value-list on TOP-LEVEL commas — commas NOT
/// inside a `<|"|>...<|"|>` string-quote span AND not nested inside
/// `{...}`/`[...]` containers (iter-231b: the template's `format_argument`
/// macro renders structured arguments, so the parser must be
/// depth-aware; the pre-iter-231b string-only tracker mangled any nested
/// value containing a comma).
fn split_gemma4_top_level(s: &str) -> Vec<&str> {
    let mut out = Vec::new();
    let mut start = 0usize;
    let mut in_str = false;
    let mut depth: usize = 0;
    let bytes = s.as_bytes();
    let mut i = 0usize;
    while i < bytes.len() {
        // Detect `<|"|>` (5 bytes) at byte i (toggles string state).
        if bytes[i..].starts_with(b"<|\"|>") {
            in_str = !in_str;
            i += 5;
            continue;
        }
        if !in_str {
            match bytes[i] {
                b'{' | b'[' => depth += 1,
                b'}' | b']' => depth = depth.saturating_sub(1),
                b',' if depth == 0 => {
                    out.push(&s[start..i]);
                    start = i + 1;
                }
                _ => {}
            }
        }
        i += 1;
    }
    if start < s.len() {
        out.push(&s[start..]);
    }
    out
}

/// Split a Gemma 4 kv pair on its TOP-LEVEL `:` — the first colon that is
/// neither inside a `<|"|>...<|"|>` string span nor inside nested
/// `{...}`/`[...]` containers.  (Keys are bare identifiers, but a naive
/// `split_once(':')` is still correct for them; this scanner exists so a
/// STRING value containing `:` — e.g. a URL — can never confuse the
/// boundary when the caller is reused on nested pairs whose values lead
/// with a container.)
fn split_gemma4_kv_once(s: &str) -> Option<(&str, &str)> {
    let mut in_str = false;
    let mut depth: usize = 0;
    let bytes = s.as_bytes();
    let mut i = 0usize;
    while i < bytes.len() {
        if bytes[i..].starts_with(b"<|\"|>") {
            in_str = !in_str;
            i += 5;
            continue;
        }
        if !in_str {
            match bytes[i] {
                b'{' | b'[' => depth += 1,
                b'}' | b']' => depth = depth.saturating_sub(1),
                b':' if depth == 0 => return Some((&s[..i], &s[i + 1..])),
                _ => {}
            }
        }
        i += 1;
    }
    None
}

/// Recursively decode a Gemma 4 `format_argument`-rendered value into a
/// `serde_json::Value` (iter-231b).
///
///   * `<|"|>...<|"|>`            → `Value::String`
///   * `{k:v,...}`                → `Value::Object` (bare keys, recursive values)
///   * `[v,...]`                  → `Value::Array` (recursive items)
///   * integer / float / bool / null literals → corresponding `Value`
///   * anything else              → `Value::String` fallback (forward the
///     model's intent rather than dropping the field — pre-iter-231b
///     contract for bare unknown values).
fn gemma4_value_to_json(v: &str) -> serde_json::Value {
    let v = v.trim();
    if let Some(stripped) = v
        .strip_prefix("<|\"|>")
        .and_then(|s| s.strip_suffix("<|\"|>"))
    {
        return serde_json::Value::String(stripped.to_string());
    }
    if v.len() >= 2 && v.starts_with('{') && v.ends_with('}') {
        let inner = &v[1..v.len() - 1];
        let mut map = serde_json::Map::new();
        if !inner.trim().is_empty() {
            for kv in split_gemma4_top_level(inner) {
                match split_gemma4_kv_once(kv) {
                    Some((k, val)) => {
                        let key = k.trim();
                        if key.is_empty() {
                            return serde_json::Value::String(v.to_string());
                        }
                        map.insert(key.to_string(), gemma4_value_to_json(val));
                    }
                    // Malformed pair (no colon): preserve the raw text
                    // rather than dropping the field (fallback contract).
                    None => return serde_json::Value::String(v.to_string()),
                }
            }
        }
        return serde_json::Value::Object(map);
    }
    if v.len() >= 2 && v.starts_with('[') && v.ends_with(']') {
        let inner = &v[1..v.len() - 1];
        if inner.trim().is_empty() {
            return serde_json::Value::Array(Vec::new());
        }
        return serde_json::Value::Array(
            split_gemma4_top_level(inner)
                .into_iter()
                .map(gemma4_value_to_json)
                .collect(),
        );
    }
    if let Ok(num) = v.parse::<i64>() {
        return serde_json::Value::from(num);
    }
    if let Ok(num) = v.parse::<f64>() {
        return serde_json::Value::from(num);
    }
    match v {
        "true" => serde_json::Value::Bool(true),
        "false" => serde_json::Value::Bool(false),
        "null" => serde_json::Value::Null,
        // Fallback: treat unquoted bare-value as a string. Better to
        // forward the model's intent than to drop the field.
        _ => serde_json::Value::String(v.to_string()),
    }
}

/// Parse Qwen 3.5/3.6's `<function=NAME>\n<parameter=key>\nval\n</parameter>\n...\n</function>` body.
///
/// Whitespace-tolerant; `<parameter=key>...</parameter>` blocks become args.
fn parse_qwen35_tool_call(body: &str) -> Option<ParsedToolCall> {
    parse_qwen35_tool_call_with_wire_kinds(body, None)
}

fn parse_qwen35_tool_call_with_wire_kinds(
    body: &str,
    wire_kinds: Option<&ToolArgumentWireKinds>,
) -> Option<ParsedToolCall> {
    let body = body.trim();
    // Expect `<function=NAME>...</function>`.
    let after_func = body.strip_prefix("<function=")?;
    let name_end = after_func.find('>')?;
    let name = after_func[..name_end].trim().to_string();
    // iter-219b (2026-05-01): reject special-token-polluted names. See the
    // gemma4 sibling for the failure-mode rationale; same OpenAI-spec
    // identifier contract applies to qwen35's `<function=NAME>` extraction.
    if !is_valid_tool_name(&name) {
        return None;
    }
    let after_name_close = &after_func[name_end + 1..];
    let func_close = after_name_close.rfind("</function>")?;
    let inner = after_name_close[..func_close].trim();

    // Walk `<parameter=KEY>VAL</parameter>` blocks.
    let mut args = serde_json::Map::new();
    let mut cursor = 0usize;
    while cursor < inner.len() {
        let Some(rel_open) = inner[cursor..].find("<parameter=") else {
            break;
        };
        let p_open = cursor + rel_open;
        let key_start = p_open + "<parameter=".len();
        let Some(rel_gt) = inner[key_start..].find('>') else {
            break;
        };
        let key_end = key_start + rel_gt;
        let key = inner[key_start..key_end].trim().to_string();
        let val_start = key_end + 1;
        let Some(rel_close) = inner[val_start..].find("</parameter>") else {
            break;
        };
        let val_end = val_start + rel_close;
        let val_raw = inner[val_start..val_end].trim();
        let json_val = decode_qwen35_parameter_value(&name, &key, val_raw, wire_kinds)?;
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
/// THE factory for engine-path `ReasoningSplitter` construction
/// (ADR-005 iter-230 B, gate-1 #10). Every generate/stream/embed path
/// builds its splitter here, passing the request's
/// `SamplingParams.reasoning_forced_open`, so a construction site can't
/// silently drop the forced-open seed. Direct
/// `ReasoningSplitter::from_registration*` calls outside registry.rs
/// are grep-pinned away by `iter230_b2_factory_only_construction`.
pub fn make_reasoning_splitter(
    reg: &ModelRegistration,
    forced_open: bool,
) -> Option<ReasoningSplitter> {
    ReasoningSplitter::from_registration_forced(reg, forced_open)
}

/// Does the rendered chat prompt end inside an OPEN reasoning block
/// (ADR-005 iter-230 B decision 1)?
///
/// Rule: the trimmed prompt ends with the registration's reasoning-open
/// marker. The chat template always appends the generation prompt AFTER
/// user content, so the tail is template-controlled — not client-
/// spoofable. Truth table (pinned in tests):
///   * qwen thinking-on  → ends `<think>`      → TRUE
///   * qwen thinking-off → ends `</think>` + ws → FALSE (suppressor)
///   * Gemma thinking-on → ends `<|turn>model`  → FALSE (model emits its
///     own open marker in the completion)
///   * Gemma thinking-off → ends `<channel|>`   → FALSE (closed seed)
pub fn prompt_seeds_reasoning_open(rendered: &str, reg: &ModelRegistration) -> bool {
    match reg.reasoning_open {
        // Both sides trimmed: markers may carry a trailing newline
        // (Gemma's `<|channel>thought\n`), which `trim_end()` on the
        // prompt side would otherwise make unmatchable.
        Some(open) if !open.trim_end().is_empty() => rendered.trim_end().ends_with(open.trim_end()),
        _ => false,
    }
}

pub fn split_full_output(reg: &ModelRegistration, full_text: &str) -> (String, Option<String>) {
    split_full_output_forced(reg, full_text, false)
}

/// `split_full_output` with the iter-230 forced-open seed. The
/// non-streaming assembly path passes the request's
/// `reasoning_forced_open` so a prompt-seeded reasoning span lands in
/// `reasoning_content` instead of leaking (with its close marker) into
/// `content`.
pub fn split_full_output_forced(
    reg: &ModelRegistration,
    full_text: &str,
    forced_open: bool,
) -> (String, Option<String>) {
    let mut splitter = match make_reasoning_splitter(reg, forced_open) {
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
    (
        content,
        if reasoning.is_empty() {
            None
        } else {
            Some(reasoning)
        },
    )
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
/// - `TooManyRequiredKeys` — schema's `required` array exceeds the 12-key
///   cap; the O(2^N) permutation grammar would be unreasonably large.
/// - `UnsupportedSchemaFeature` — a nested schema uses a feature the
///   recursive compiler cannot enforce exactly (for example `not`,
///   tuple-form `items`, or depth > 32). The 400 names the feature and path.
///
/// # ADR-005 Wave-2.7 design note
/// The 12-key cap is shared with json_schema.rs. It bounds the exponential
/// subset grammar at 4096 states while admitting the nine-key r2c ReviewLens
/// contract.
///
/// # iter-231a/3.7 note
/// The wave-2.5 `UnsupportedSchema` variant (B3 — hard rejection of
/// `array`/`object` parameter types) is REMOVED: structured parameters
/// compile to recursive value rules (iter-231a permissive free-form,
/// iter-231b full-fidelity declared structure) instead of failing the
/// entire request with HTTP 400.  Deleted as a unit per the no-fallback
/// mantra — no call site may re-introduce a scalar-only gate.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EmitterError {
    /// `required` array length exceeds `MAX_REQUIRED_KEYS` (12).
    TooManyRequiredKeys { fn_name: String, count: usize },
    /// Nested schema feature the iter-231b recursive compiler cannot
    /// enforce — see variant docs above.
    UnsupportedSchemaFeature {
        fn_name: String,
        param_path: String,
        feature: String,
    },
}

/// Hard cap on the number of required keys for the permutation grammar.
/// Aligned with json_schema.rs ANY_ORDER_MAX_REQUIRED = 12 (ADR-052).
/// Larger schemas return `EmitterError::TooManyRequiredKeys`.
const MAX_REQUIRED_KEYS: usize = 12;

/// iter-231b — recursion depth cap for nested parameter schemas.  JSON
/// Schema cannot self-reference without `$ref` (rejected as an
/// unsupported feature), so nesting is finite; this cap bounds the
/// emitted grammar size against pathological deep schemas.
const MAX_NESTED_DEPTH: usize = 32;

/// iter-231b — property-count cap per nested object, mirroring
/// json_schema.rs's 32-property bound (its `build_unified_inner` has the
/// same shape).  Larger objects return
/// `EmitterError::UnsupportedSchemaFeature`.
const MAX_NESTED_PROPERTIES: usize = 32;

fn combine_schema_variant_grammars(grammars: Vec<String>) -> Result<String, String> {
    if grammars.len() == 1 {
        return Ok(grammars.into_iter().next().expect("one grammar"));
    }
    use crate::serve::api::grammar::{parser::parse_generated, rename_rules, serialize};

    let mut roots = Vec::with_capacity(grammars.len());
    let mut bodies = String::new();
    for (index, source) in grammars.iter().enumerate() {
        let parsed = parse_generated(source)
            .map_err(|error| format!("schema variant {index} grammar failed to parse: {error}"))?;
        let renamed = rename_rules(&parsed, |name| format!("variant-{index}-{name}"));
        roots.push(format!("variant-{index}-root"));
        bodies.push_str(&serialize(&renamed));
    }
    let combined = format!("root ::= {}\n{}", roots.join(" | "), bodies);
    parse_generated(&combined)
        .map_err(|error| format!("combined schema variants failed to parse: {error}"))?;
    Ok(combined)
}

/// JSON object keys are strings by construction.  OpenCode plugin tools built
/// from a Zod `record(string, unknown)` nevertheless emit the redundant JSON
/// Schema keyword `propertyNames: {"type":"string"}`.  Treat only the
/// semantically-unconstrained forms as no-ops; a pattern/enum/length-constrained
/// key schema still fails closed because the wildcard object-key grammar cannot
/// enforce it.
fn validate_unconstrained_property_names(
    fn_name: &str,
    path: &str,
    schema: Option<&serde_json::Value>,
) -> Result<(), EmitterError> {
    let Some(schema) = schema else {
        return Ok(());
    };
    let unconstrained = match schema {
        serde_json::Value::Bool(true) => true,
        serde_json::Value::Object(obj) if obj.is_empty() => true,
        serde_json::Value::Object(obj) => {
            obj.len() == 1
                && matches!(obj.get("type"), Some(serde_json::Value::String(kind)) if kind == "string")
        }
        _ => false,
    };
    if unconstrained {
        return Ok(());
    }
    Err(EmitterError::UnsupportedSchemaFeature {
        fn_name: fn_name.to_string(),
        param_path: path.to_string(),
        feature: "constrained propertyNames".to_string(),
    })
}

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
            EmitterError::UnsupportedSchemaFeature {
                fn_name,
                param_path,
                feature,
            } => write!(
                f,
                "function '{}' parameter '{}' uses unsupported schema feature \
                 '{}'; the iter-231b nested compiler supports scalars, enums, \
                 type-unions, anyOf/oneOf, object (properties/required<=8/\
                 additionalProperties) and array (single-schema items); \
                 rewrite the parameter schema or drop the feature",
                fn_name, param_path, feature
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

fn gemma4_enum_wire(value: &serde_json::Value) -> Result<String, String> {
    match value {
        serde_json::Value::String(value) => Ok(format!("<|\"|>{value}<|\"|>")),
        serde_json::Value::Number(_) | serde_json::Value::Bool(_) | serde_json::Value::Null => {
            serde_json::to_string(value).map_err(|error| error.to_string())
        }
        serde_json::Value::Array(values) => Ok(format!(
            "[{}]",
            values
                .iter()
                .map(gemma4_enum_wire)
                .collect::<Result<Vec<_>, _>>()?
                .join(",")
        )),
        serde_json::Value::Object(values) => {
            let mut fields = Vec::with_capacity(values.len());
            for (name, value) in values {
                if name.is_empty()
                    || name.chars().any(|character| {
                        matches!(character, ',' | ':' | '{' | '}' | '[' | ']' | '<')
                    })
                {
                    return Err(format!(
                        "container enum has an unrepresentable Gemma key {name:?}"
                    ));
                }
                fields.push(format!("{name}:{}", gemma4_enum_wire(value)?));
            }
            Ok(format!("{{{}}}", fields.join(",")))
        }
    }
}

#[derive(Default)]
struct GemmaBareKeyTrie {
    terminal: bool,
    children: std::collections::BTreeMap<char, GemmaBareKeyTrie>,
}

fn gemma4_bare_key_excluding_gbnf(excluded: &[String]) -> Result<String, String> {
    let mut trie = GemmaBareKeyTrie::default();
    let mut total_characters = 0usize;
    for name in excluded {
        if name.is_empty()
            || name.chars().any(|character| {
                character.is_control()
                    || matches!(character, ',' | ':' | '{' | '}' | '[' | ']' | '<')
            })
        {
            return Err(format!(
                "declared property {name:?} cannot be represented as an exact Gemma bare key"
            ));
        }
        let characters = name.chars().count();
        total_characters = total_characters.saturating_add(characters);
        if characters > 256 || total_characters > 4096 {
            return Err(
                "declared property names exceed the exact Gemma wildcard exclusion budget (256 characters per key, 4096 total)"
                    .to_string(),
            );
        }
        let mut node = &mut trie;
        for character in name.chars() {
            node = node.children.entry(character).or_default();
        }
        node.terminal = true;
    }
    Ok(gemma4_bare_key_trie_body(&trie, true))
}

fn gemma4_bare_key_trie_body(node: &GemmaBareKeyTrie, root: bool) -> String {
    let mut alternatives = Vec::new();
    if !root && !node.terminal {
        alternatives.push(gbnf_literal(""));
    }
    for (character, child) in &node.children {
        alternatives.push(format!(
            "{} {}",
            gbnf_literal(&character.to_string()),
            gemma4_bare_key_trie_body(child, false)
        ));
    }
    let mut mismatch = String::from(r#"[^,:{}\[\]<"#);
    for character in node.children.keys() {
        match character {
            '[' | ']' | '-' | '\\' => {
                mismatch.push('\\');
                mismatch.push(*character);
            }
            _ => mismatch.push(*character),
        }
    }
    mismatch.push(']');
    alternatives.push(format!("{mismatch} gemma4-json-key*"));
    format!("( {} )", alternatives.join(" | "))
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
/// iter-231a (B3 supersession): `array` and `object` types compile via the
/// iter-231b recursive nested-schema converter (`gemma4_nested_value_rule`).
/// The wave-2.5 B3 hard rejection (`EmitterError::UnsupportedSchema` →
/// HTTP 400) made every tools-bearing request fail against tool schemas
/// with free-form object parameters (MCP servers, nested schemas).
fn gemma4_value_gbnf(
    fn_name: &str,
    param_name: &str,
    schema: &serde_json::Value,
    rules: &mut Vec<(String, String)>,
    rule_counter: &mut u32,
) -> Result<String, EmitterError> {
    if schema == &serde_json::Value::Bool(false) {
        return Ok(r#"[^\U00000000-\U0010FFFF]"#.to_string());
    }
    let obj = match schema.as_object() {
        Some(o) => o,
        None => {
            // Untyped / unknown → accept any Gemma value (string, bare
            // scalar, or structured kv-list value).
            return Ok(GEMMA4_TOP_ANY_VAL.to_string());
        }
    };

    for keyword in ["anyOf", "oneOf"] {
        if let Some(serde_json::Value::Array(branches)) = obj.get(keyword) {
            let mut alternatives = Vec::with_capacity(branches.len());
            for (index, branch) in branches.iter().enumerate() {
                alternatives.push(gemma4_value_gbnf(
                    fn_name,
                    &format!("{param_name}-{keyword}-{index}"),
                    branch,
                    rules,
                    rule_counter,
                )?);
            }
            return Ok(format!("( {} )", alternatives.join(" | ")));
        }
    }

    // Every finite value is rendered through Gemma's native recursive wire
    // syntax; filtering only strings would silently widen scalar/container
    // enums through the type fallback below.
    if let Some(serde_json::Value::Array(values)) = obj.get("enum") {
        if values.is_empty() {
            return Ok(r#"[^\U00000000-\U0010FFFF]"#.to_string());
        }
        let alts = values
            .iter()
            .map(|value| {
                gemma4_enum_wire(value)
                    .map(|wire| gbnf_literal(&wire))
                    .map_err(|feature| EmitterError::UnsupportedSchemaFeature {
                        fn_name: fn_name.to_string(),
                        param_path: param_name.to_string(),
                        feature,
                    })
            })
            .collect::<Result<Vec<_>, _>>()?;
        return Ok(format!("( {} )", alts.join(" | ")));
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
            //
            // iter-231c: `pattern` replaces the generic char* body with the
            // compiled regex between the markers.
            match compile_string_assertion(
                fn_name,
                &format!("/{}", param_name),
                obj,
                crate::serve::api::grammar::regex_gbnf::Surface::GemmaMarkerString,
            )? {
                Some(body) => Ok(format!(
                    "{} {} {}",
                    gbnf_literal("<|\"|>"),
                    body,
                    gbnf_literal("<|\"|>")
                )),
                None => match bounded_string_body(obj, "gemma4-str-char") {
                    Some(body) => Ok(format!(
                        "{} {} {}",
                        gbnf_literal("<|\"|>"),
                        body,
                        gbnf_literal("<|\"|>")
                    )),
                    None => Ok("gemma4-str-val".to_string()),
                },
            }
        }
        "integer" => Ok(
            compile_integer_assertion(fn_name, &format!("/{}", param_name), obj)?
                .unwrap_or_else(|| "gemma4-int-val".to_string()),
        ),
        "number" => Ok("gemma4-num-val".to_string()),
        "boolean" => Ok("gemma4-bool-val".to_string()),
        "null" => Ok("gemma4-null-val".to_string()),
        // iter-231b: structured types → full-fidelity recursive compiler.
        "array" | "object" => gemma4_nested_value_rule(
            fn_name,
            &format!("/{}", param_name),
            schema,
            rules,
            rule_counter,
            1,
        ),
        _ => {
            // Unrecognised type string — accept any value (conservative).
            Ok(GEMMA4_TOP_ANY_VAL.to_string())
        }
    }
}

/// Top-level value body for untyped / unknown-typed Gemma parameters:
/// the template's `format_argument` renders strings (`<|"|>…<|"|>`),
/// bare scalars, AND structured kv-list values, so all are accepted.
const GEMMA4_TOP_ANY_VAL: &str = "( gemma4-any-val | gemma4-json-obj | gemma4-json-arr )";

/// iter-231b — recursive nested-schema compiler for Gemma 4 tool
/// parameters (`format_argument` kv surface: strings `<|"|>…<|"|>`, bare
/// keys at every nesting level via `escape_keys=False`, comma-separated,
/// no whitespace).
///
/// Same contract as `qwen35_nested_value_rule` — constrain everything the
/// schema DECLARES, stay open where the schema is open:
///
///   * scalars / enums / type-unions / anyOf / oneOf — exact bodies.
///   * `object` with `properties` — declared keys with per-key value
///     grammars; `required` (≤12) enforced any-order via permutation;
///     optional keys follow in a Kleene suffix (top-level contract).
///     `additionalProperties:false` closes the key set; unset/true adds
///     a wildcard kv tail.
///   * `object`/`array` with NO declared shape → the permissive
///     recursive `gemma4-json-obj` / `gemma4-json-arr` rules.
///   * `array` with `items` → typed item grammar; absent → permissive.
///
/// Same `EmitterError::UnsupportedSchemaFeature` rejection set and the
/// same rejection-strength contract as the Qwen compiler (required keys
/// + closed objects are strictly enforced; open objects carry the
/// wildcard extra-kv caveat — see `qwen35_nested_value_rule`).
fn gemma4_nested_value_rule(
    fn_name: &str,
    path: &str,
    schema: &serde_json::Value,
    rules: &mut Vec<(String, String)>,
    rule_counter: &mut u32,
    depth: usize,
) -> Result<String, EmitterError> {
    if depth > MAX_NESTED_DEPTH {
        return Err(EmitterError::UnsupportedSchemaFeature {
            fn_name: fn_name.to_string(),
            param_path: path.to_string(),
            feature: format!("nesting depth > {}", MAX_NESTED_DEPTH),
        });
    }
    if schema == &serde_json::Value::Bool(false) {
        return Ok(r#"[^\U00000000-\U0010FFFF]"#.to_string());
    }
    let obj = match schema.as_object() {
        Some(o) => o,
        None => return Ok("gemma4-json-val".to_string()),
    };

    validate_unconstrained_property_names(fn_name, path, obj.get("propertyNames"))?;

    for feat in [
        "allOf",
        "$ref",
        "$defs",
        "not",
        "if",
        "then",
        "else",
        "dependentSchemas",
        "patternProperties",
        "contains",
    ] {
        if obj.contains_key(feat) {
            return Err(EmitterError::UnsupportedSchemaFeature {
                fn_name: fn_name.to_string(),
                param_path: path.to_string(),
                feature: feat.to_string(),
            });
        }
    }

    for comb in ["anyOf", "oneOf"] {
        if let Some(serde_json::Value::Array(subs)) = obj.get(comb) {
            if subs.is_empty() {
                return Err(EmitterError::UnsupportedSchemaFeature {
                    fn_name: fn_name.to_string(),
                    param_path: path.to_string(),
                    feature: format!("empty {}", comb),
                });
            }
            let mut alts: Vec<String> = Vec::with_capacity(subs.len());
            for (i, s) in subs.iter().enumerate() {
                alts.push(gemma4_nested_value_rule(
                    fn_name,
                    &format!("{}/{}/{}", path, comb, i),
                    s,
                    rules,
                    rule_counter,
                    depth + 1,
                )?);
            }
            return Ok(format!("( {} )", alts.join(" | ")));
        }
    }

    // enum → exact native Gemma wire literals, including containers.
    if let Some(serde_json::Value::Array(values)) = obj.get("enum") {
        if values.is_empty() {
            return Ok(r#"[^\U00000000-\U0010FFFF]"#.to_string());
        }
        let alts = values
            .iter()
            .map(|value| {
                gemma4_enum_wire(value)
                    .map(|wire| gbnf_literal(&wire))
                    .map_err(|feature| EmitterError::UnsupportedSchemaFeature {
                        fn_name: fn_name.to_string(),
                        param_path: path.to_string(),
                        feature,
                    })
            })
            .collect::<Result<Vec<_>, _>>()?;
        return Ok(format!("( {} )", alts.join(" | ")));
    }

    match obj.get("type") {
        None => Ok("gemma4-json-val".to_string()),
        Some(serde_json::Value::Array(types)) => {
            let mut alts: Vec<String> = Vec::with_capacity(types.len());
            for (i, t) in types.iter().enumerate() {
                let Some(tstr) = t.as_str() else {
                    return Err(EmitterError::UnsupportedSchemaFeature {
                        fn_name: fn_name.to_string(),
                        param_path: path.to_string(),
                        feature: "non-string type union entry".to_string(),
                    });
                };
                let mut stub = serde_json::Map::new();
                stub.insert("type".into(), serde_json::Value::String(tstr.into()));
                alts.push(gemma4_nested_value_rule(
                    fn_name,
                    &format!("{}/type/{}", path, i),
                    &serde_json::Value::Object(stub),
                    rules,
                    rule_counter,
                    depth + 1,
                )?);
            }
            Ok(format!("( {} )", alts.join(" | ")))
        }
        Some(serde_json::Value::String(t)) => match t.as_str() {
            "string" => {
                // iter-231c: `pattern` constrains the marker-string content.
                match compile_string_assertion(
                    fn_name,
                    path,
                    obj,
                    crate::serve::api::grammar::regex_gbnf::Surface::GemmaMarkerString,
                )? {
                    Some(body) => Ok(format!(
                        "{} {} {}",
                        gbnf_literal("<|\"|>"),
                        body,
                        gbnf_literal("<|\"|>")
                    )),
                    None => match bounded_string_body(obj, "gemma4-str-char") {
                        Some(body) => Ok(format!(
                            "{} {} {}",
                            gbnf_literal("<|\"|>"),
                            body,
                            gbnf_literal("<|\"|>")
                        )),
                        None => Ok("gemma4-str-val".to_string()),
                    },
                }
            }
            "integer" => Ok(compile_integer_assertion(fn_name, path, obj)?
                .unwrap_or_else(|| "gemma4-int-val".to_string())),
            "number" => Ok("gemma4-num-val".to_string()),
            "boolean" => Ok("gemma4-bool-val".to_string()),
            "null" => Ok("gemma4-null-val".to_string()),
            "object" => gemma4_nested_object(fn_name, path, obj, rules, rule_counter, depth),
            "array" => gemma4_nested_array(fn_name, path, obj, rules, rule_counter, depth),
            other => Err(EmitterError::UnsupportedSchemaFeature {
                fn_name: fn_name.to_string(),
                param_path: path.to_string(),
                feature: format!("type '{}'", other),
            }),
        },
        Some(_) => Err(EmitterError::UnsupportedSchemaFeature {
            fn_name: fn_name.to_string(),
            param_path: path.to_string(),
            feature: "non-string type".to_string(),
        }),
    }
}

/// iter-231b — nested `object` compiler (Gemma kv surface).  Keys are
/// BARE literals (the template invokes `format_argument` with
/// `escape_keys=False` recursively at every nesting level).
fn gemma4_nested_object(
    fn_name: &str,
    path: &str,
    obj: &serde_json::Map<String, serde_json::Value>,
    rules: &mut Vec<(String, String)>,
    rule_counter: &mut u32,
    depth: usize,
) -> Result<String, EmitterError> {
    let properties = obj.get("properties").and_then(|p| p.as_object());
    let additional_closed = matches!(
        obj.get("additionalProperties"),
        Some(serde_json::Value::Bool(false))
    );
    let additional_schema = obj.get("additionalProperties");

    let props = match properties {
        Some(p) if !p.is_empty() => p,
        _ => {
            if additional_closed {
                return Ok(r#""{" "}""#.to_string());
            }
            if additional_schema.is_some_and(serde_json::Value::is_object) {
                let value = gemma4_nested_value_rule(
                    fn_name,
                    &format!("{path}/additionalProperties"),
                    additional_schema.expect("present"),
                    rules,
                    rule_counter,
                    depth + 1,
                )?;
                return Ok(format!(
                    r#""{{" ( "}}" | gemma4-json-key ":" {value} ( "," gemma4-json-key ":" {value} )* "}}" )"#
                ));
            }
            return Ok("gemma4-json-obj".to_string());
        }
    };

    let required_set: std::collections::HashSet<&str> = obj
        .get("required")
        .and_then(|v| v.as_array())
        .map(|arr| arr.iter().filter_map(|v| v.as_str()).collect())
        .unwrap_or_default();

    if required_set.len() > MAX_REQUIRED_KEYS {
        return Err(EmitterError::TooManyRequiredKeys {
            fn_name: format!("{} (nested {})", fn_name, path),
            count: required_set.len(),
        });
    }
    if props.len() > MAX_NESTED_PROPERTIES {
        return Err(EmitterError::UnsupportedSchemaFeature {
            fn_name: fn_name.to_string(),
            param_path: path.to_string(),
            feature: format!("> {} properties", MAX_NESTED_PROPERTIES),
        });
    }

    let mut req_kv: Vec<String> = Vec::new();
    let mut opt_kv: Vec<String> = Vec::new();
    let mut sorted_keys: Vec<&String> = props.keys().collect();
    sorted_keys.sort();
    for key in sorted_keys {
        let val_body = gemma4_nested_value_rule(
            fn_name,
            &format!("{}/properties/{}", path, key),
            &props[key],
            rules,
            rule_counter,
            depth + 1,
        )?;
        *rule_counter += 1;
        let kv_name = format!("g4n-{}-kv", *rule_counter);
        rules.push((
            kv_name.clone(),
            format!("{} \":\" {}", gbnf_literal(key), val_body),
        ));
        if required_set.contains(key.as_str()) {
            req_kv.push(kv_name);
        } else {
            opt_kv.push(kv_name);
        }
    }

    // Wildcard kv for open objects: an exact key complement prevents a
    // declared optional key from bypassing its own value schema.
    let extra_kv: Option<String> = if additional_closed {
        None
    } else {
        let names = props.keys().cloned().collect::<Vec<_>>();
        let key_body = gemma4_bare_key_excluding_gbnf(&names).map_err(|feature| {
            EmitterError::UnsupportedSchemaFeature {
                fn_name: fn_name.to_string(),
                param_path: path.to_string(),
                feature,
            }
        })?;
        let value_body = match additional_schema {
            Some(schema) => gemma4_nested_value_rule(
                fn_name,
                &format!("{path}/additionalProperties"),
                schema,
                rules,
                rule_counter,
                depth + 1,
            )?,
            None => "gemma4-json-val".to_string(),
        };
        *rule_counter += 1;
        let name = format!("g4n-{}-extra-kv", *rule_counter);
        rules.push((name.clone(), format!("{key_body} \":\" {value_body}")));
        Some(name)
    };

    build_nested_obj_body(
        "g4n",
        req_kv,
        opt_kv,
        extra_kv,
        r#"",""#,
        rules,
        rule_counter,
    )
    .map(|inner| format!(r#""{{" {} "}}""#, inner))
}

/// iter-231b — nested `array` compiler (Gemma kv surface).
fn gemma4_nested_array(
    fn_name: &str,
    path: &str,
    obj: &serde_json::Map<String, serde_json::Value>,
    rules: &mut Vec<(String, String)>,
    rule_counter: &mut u32,
    depth: usize,
) -> Result<String, EmitterError> {
    let min = obj
        .get("minItems")
        .and_then(serde_json::Value::as_u64)
        .unwrap_or(0);
    let max = obj.get("maxItems").and_then(serde_json::Value::as_u64);
    match obj.get("items") {
        None if min == 0 && max.is_none() => Ok("gemma4-json-arr".to_string()),
        None => Ok(bounded_array_body("gemma4-json-val", r#"",""#, min, max)),
        Some(serde_json::Value::Object(_)) | Some(serde_json::Value::Bool(_)) => {
            let item_rule = gemma4_nested_value_rule(
                fn_name,
                &format!("{}/items", path),
                obj.get("items").expect("items checked above"),
                rules,
                rule_counter,
                depth + 1,
            )?;
            Ok(bounded_array_body(&item_rule, r#"",""#, min, max))
        }
        Some(serde_json::Value::Array(_)) => Err(EmitterError::UnsupportedSchemaFeature {
            fn_name: fn_name.to_string(),
            param_path: path.to_string(),
            feature: "tuple-form items".to_string(),
        }),
        Some(_) => Err(EmitterError::UnsupportedSchemaFeature {
            fn_name: fn_name.to_string(),
            param_path: path.to_string(),
            feature: "non-object items".to_string(),
        }),
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
/// Kleene-star suffix. Hard cap `MAX_REQUIRED_KEYS` (12) bounds O(2^N)
/// grammar blowup.
///
/// # Structured parameters (iter-231a, supersedes Wave 2.5 B3)
///
/// Parameters typed `array` or `object` compile to the permissive
/// recursive `gemma4-json-arr` / `gemma4-json-obj` rules — contents
/// unconstrained, structure constrained.  (Wave 2.5 B3 rejected these
/// with `EmitterError::UnsupportedSchema` → HTTP 400; that gate broke
/// every tool schema carrying free-form object params, e.g. MCP tools.)
fn gemma4_tool_call_gbnf(
    fn_name: &str,
    params_schema: &serde_json::Value,
    shape: GrammarShape,
) -> Result<String, EmitterError> {
    if let Some(obj) = params_schema.as_object() {
        validate_unconstrained_property_names(fn_name, "/", obj.get("propertyNames"))?;
    }
    let mut rules: Vec<(String, String)> = Vec::new();
    let mut rule_counter: u32 = 0;

    // Gemma 4 primitive rules (shared across all function grammars).
    //
    // String: `<|"|>` any-safe-char* `<|"|>`. Source-shaped tool arguments
    // routinely contain a bare `<` (`Vec<T>`, `Formatter<'_>`, comparisons),
    // so rejecting every `<` truncates otherwise valid calls at the sampler.
    // The staged alternatives below accept `<` and every partial delimiter
    // prefix while reserving only the exact five-byte `<|"|>` terminator.
    // A complete delimiter cannot be represented literally inside this
    // marker-string surface; that is a limitation of the model format itself.
    rules.push((
        "gemma4-str-char".to_string(),
        r##"[^<\\] | [\\] [^\x00-\x1F] | "<" [^|] | "<|" [^"] | "<|\"" [^|] | "<|\"|" [^>]"##
            .to_string(),
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
        r#""-"? ([0] | [1-9] [0-9]{0,15}) ("." [0-9]{1,16})? ([eE] [-+]? [0-9]{1,16})?"#
            .to_string(),
    ));
    rules.push((
        "gemma4-bool-val".to_string(),
        r#""true" | "false""#.to_string(),
    ));
    rules.push(("gemma4-null-val".to_string(), r#""null""#.to_string()));
    rules.push((
        "gemma4-any-val".to_string(),
        r#"gemma4-str-val | gemma4-num-val | gemma4-bool-val | gemma4-null-val"#.to_string(),
    ));

    // iter-231a (B3 supersession): permissive recursive kv-list rules for
    // `object` / `array` parameters.  The chat template's
    // `format_argument(value, escape_keys=False)` macro
    // (test_fixtures/gemma4-apex-embedded-chat-template.jinja:111-140,
    // invoked from :198 with `escape_keys=False` propagated recursively)
    // renders structured args as:
    //   * object  → `{key:val,...}`   — keys BARE at every nesting level
    //   * array   → `[val,...]`
    //   * string  → `<|"|>s<|"|>`     (= gemma4-str-val)
    //   * bool/num/null → bare        (= existing scalar rules)
    //
    // `gemma4-json-key` matches bare keys: any run of chars that cannot
    // collide with the kv-list structure (`,` `:` `{` `}` `[` `]`) or the
    // `<tool_call|>` close marker (`<`) — same conservative-charset
    // contract as `gemma4-str-char`.
    //
    // Same supersession rationale as the Qwen emitter: the grammar still
    // constrains call structure (function name, declared param keys,
    // kv-list layout, scalar types) but treats the CONTENTS of
    // object/array parameters as "any well-formed kv-list value", ending
    // the wave-2.5 `EmitterError::UnsupportedSchema` → HTTP 400 hard
    // failure on real-world tool schemas (MCP free-form objects).
    rules.push((
        "gemma4-json-key".to_string(),
        r#"[^,:{}\[\]<]+"#.to_string(),
    ));
    rules.push((
        "gemma4-json-obj".to_string(),
        r#""{" ("}" | gemma4-json-key ":" gemma4-json-val ("," gemma4-json-key ":" gemma4-json-val)* "}")"#.to_string(),
    ));
    rules.push((
        "gemma4-json-arr".to_string(),
        r#""[" ("]" | gemma4-json-val ("," gemma4-json-val)* "]")"#.to_string(),
    ));
    rules.push((
        "gemma4-json-val".to_string(),
        "gemma4-str-val | gemma4-num-val | gemma4-bool-val | gemma4-null-val | gemma4-json-obj | gemma4-json-arr".to_string(),
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

    // Build the per-call body (shape-independent).  The `single_body`
    // string is the GBNF expression that matches exactly one Gemma 4
    // tool-call body (`call:NAME{kv-list}`), with no enclosing markers.
    // The shape selector below wraps it in either the body-only
    // (SingleBody) or marker-wrapped + repeat (OneOrMoreCalls) shape.
    let single_body: String = if let Some(props) = properties {
        if props.is_empty() {
            format!("{} {} {}", prefix_lit, open_lit, close_lit)
        } else {
            let mut required_kv_names: Vec<String> = Vec::new();
            let mut optional_kv_names: Vec<String> = Vec::new();
            let mut sorted_keys: Vec<&String> = props.keys().collect();
            sorted_keys.sort();
            for key in &sorted_keys {
                let val_schema = &props[*key];
                // iter-231a: array/object map to permissive json rules.
                let val_rule = gemma4_value_gbnf(
                    fn_name,
                    key.as_str(),
                    val_schema,
                    &mut rules,
                    &mut rule_counter,
                )?;
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

            format!("{} {} {} {}", prefix_lit, open_lit, kv_body_rule, close_lit)
        }
    } else {
        rules.push(("gemma4-any-kv-char".to_string(), r#"[^}]"#.to_string()));
        format!(
            "{} {} gemma4-any-kv-char* {}",
            prefix_lit, open_lit, close_lit
        )
    };

    // Wave 2.7 W-η + Wave 3.5 HIGH-1: select root rule per `shape`.
    //
    //   * `SingleBody`                       → body-only root (single
    //     call, no markers; legacy lazy path).
    //   * `OneOrMoreCalls{p}`                → wrap body with Gemma 4
    //     marker pair `<|tool_call>` / `<tool_call|>` (verbatim from
    //     models/gemma4/chat_template.jinja:189-205) and repeat per
    //     `parallel`.  No separator: chat_template emits calls 2+
    //     immediately after the previous call's close marker.
    //   * `OneOrMoreCallsBodyOnly{p}`        → Wave 3.5 HIGH-1 Auto-lazy
    //     variant.  Strips the FIRST open marker (consumed by the
    //     awaiting_trigger no-op gate before the grammar is engaged).
    //     Single: `body close_marker`; parallel:
    //     `body close_marker ( open_marker body close_marker )* space`.
    //     See `GrammarShape::OneOrMoreCallsBodyOnly` doc-comment for the
    //     production-order rationale.
    // iter-219b grammar-exhaust fix (2026-05-01): drop trailing ` space`
    // from all root_body shapes. The `space` rule
    // (`| " " | "\n"{1,2} [ \t]{0,20}`) was inherited from the peer's
    // SPACE_RULE convention for JSON Schema grammars where it terminates
    // value rules. For tool-call grammars it served as a trailing-
    // whitespace allowance after the close marker — but its empty-alt +
    // `[ \t]{0,20}` (min 0) tail kept `is_dead` from ever flipping
    // post-close (Agent C audit 2026-05-01,
    // `docs/research/iter219b-grammar-exhaust-audit-2026-05-01.md`),
    // breaking iter-218's claimed "natural grammar-exhaust" termination.
    // Without the trailing `space`, the grammar exhausts cleanly on the
    // close marker; the engine's `is_dead`-break path at engine.rs:5045-5059
    // then fires on the next sampled byte. Per chat-template inspection
    // (regex at tokenizer_config.json: `<|tool_call>...<tool_call|><turn|>?`),
    // the model is trained to emit `<turn|>` (which is in eos_token_ids)
    // immediately after `<tool_call|>` with no separating whitespace, so
    // the trailing-whitespace allowance was never load-bearing for the
    // happy path.
    let root_body = match shape {
        GrammarShape::SingleBody => single_body.clone(),
        GrammarShape::OneOrMoreCalls { parallel } => {
            let open_marker = gbnf_literal("<|tool_call>");
            let close_marker = gbnf_literal("<tool_call|>");
            let g4_call_rule = "gemma4-call".to_string();
            rules.push((
                g4_call_rule.clone(),
                format!("{} {} {}", open_marker, single_body, close_marker),
            ));
            // `(call)+` GBNF idiom: `call call*` — at least one, optional
            // repeat.  Mirrors the peer's `p.repeat(call, 1, parallel?-1:1)`.
            if parallel {
                format!("{} {}*", g4_call_rule, g4_call_rule)
            } else {
                g4_call_rule
            }
        }
        GrammarShape::OneOrMoreCallsBodyOnly { parallel } => {
            // Wave 3.5 HIGH-1: leading open marker is stripped (consumed
            // by the awaiting_trigger gate before the grammar fires).
            // The trailing close marker IS in the grammar — those bytes
            // flow through `accept_bytes` once the runtime is eager.
            let open_marker = gbnf_literal("<|tool_call>");
            let close_marker = gbnf_literal("<tool_call|>");
            if parallel {
                // First call: body close — no leading open.  Subsequent
                // calls: full open+body+close, no inter-call separator
                // (chat_template.jinja:189-205 emits calls back-to-back).
                let g4_call_rule = "gemma4-call".to_string();
                rules.push((
                    g4_call_rule.clone(),
                    format!("{} {} {}", open_marker, single_body, close_marker),
                ));
                format!("{} {} {}*", single_body, close_marker, g4_call_rule)
            } else {
                format!("{} {}", single_body, close_marker)
            }
        }
    };
    rules.push(("root".to_string(), root_body));

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
    let rule_name = format!(
        "g4req-{}-{}",
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
/// keys in Kleene-star suffix. Hard cap `MAX_REQUIRED_KEYS` (12).
///
/// # Structured parameters (iter-231a, supersedes Wave 2.5 B3)
///
/// Parameters typed `array` or `object` compile to the permissive
/// recursive `qwen35-json-arr` / `qwen35-json-obj` rules — contents
/// unconstrained, structure constrained.  (Wave 2.5 B3 rejected these
/// with `EmitterError::UnsupportedSchema` → HTTP 400; that gate broke
/// every tool schema carrying free-form object params, e.g. MCP tools.)
fn qwen35_tool_call_gbnf(
    fn_name: &str,
    params_schema: &serde_json::Value,
    shape: GrammarShape,
) -> Result<String, EmitterError> {
    if let Some(obj) = params_schema.as_object() {
        validate_unconstrained_property_names(fn_name, "/", obj.get("propertyNames"))?;
    }
    let mut rules: Vec<(String, String)> = Vec::new();

    // Qwen 3.5/3.6 value primitives — values sit between XML tags, raw text
    // (no JSON quoting for strings).  Numbers/booleans are JSON-serialized
    // (the `tojson` Jinja filter is used in the template).
    rules.push((
        "qwen35-str-char".to_string(),
        // Raw parameters need bare `<` for source code, but an unrestricted
        // `<` branch can swallow the exact `</parameter>` terminator and then
        // accept adjacent tool calls as parameter text. These staged mismatch
        // alternatives reserve only the complete close tag. Partial prefixes
        // and close-like source text remain valid parameter bytes.
        r#"[^<\\] | [\\] [^\x00-\x1F] | "<" [^/] | "</" [^p] | "</p" [^a] | "</pa" [^r] | "</par" [^a] | "</para" [^m] | "</param" [^e] | "</parame" [^t] | "</paramet" [^e] | "</paramete" [^r] | "</parameter" [^>]"#
            .to_string(),
    ));
    rules.push(("qwen35-str-val".to_string(), "qwen35-str-char*".to_string()));
    rules.push((
        "qwen35-int-val".to_string(),
        r#""-"? ([0] | [1-9] [0-9]{0,15})"#.to_string(),
    ));
    rules.push((
        "qwen35-num-val".to_string(),
        r#""-"? ([0] | [1-9] [0-9]{0,15}) ("." [0-9]{1,16})? ([eE] [-+]? [0-9]{1,16})?"#
            .to_string(),
    ));
    rules.push((
        "qwen35-bool-val".to_string(),
        r#""true" | "false""#.to_string(),
    ));
    rules.push(("qwen35-null-val".to_string(), r#""null""#.to_string()));
    rules.push((
        "qwen35-any-val".to_string(),
        r#"qwen35-str-val | qwen35-num-val | qwen35-bool-val | qwen35-null-val"#.to_string(),
    ));

    // iter-231a (B3 supersession): permissive recursive JSON rules for
    // `object` / `array` parameters.  The chat template renders every
    // non-string argument via the `tojson` filter
    // (`tojson` — native compact JSON uses one space after commas and
    // colons). Accept both that canonical surface and its whitespace-free
    // equivalent so constrained decoding does not force the model away from
    // the syntax shown in its own prompt:
    //
    //   * `qwen35-json-char` follows JSON's normal string-character rule.
    //     A literal `<` is valid JSON and `\<` is not a valid JSON escape;
    //     the enclosing JSON structure and parameter close tag delimit it.
    //   * `qwen35-json-obj` / `qwen35-json-arr` are mutually recursive
    //     with `qwen35-json-val` (GBNF supports recursion — same idiom as
    //     json_schema.rs's `value`/`object`/`array` primitives).
    //   * Object keys are JSON strings (serde emits them quoted).
    //   * Numbers reuse `qwen35-num-val` (covers ints + floats).
    //
    // These rules are a deliberate SUPERSET of any nested schema: the
    // grammar constrains the call structure (function name, parameter
    // keys, tag layout, scalar types) but treats the CONTENTS of
    // object/array parameters as "any well-formed JSON".  Real-world
    // tool schemas (e.g. MCP servers exposing free-form `config` /
    // `arguments` objects) previously hard-failed the entire request
    // with `EmitterError::UnsupportedSchema` → HTTP 400.
    rules.push((
        "qwen35-json-char".to_string(),
        r#"[^"\\\x00-\x1F] | [\\] (["\\/bfnrt] | [u] [0-9a-fA-F]{4})"#.to_string(),
    ));
    rules.push((
        "qwen35-json-str".to_string(),
        r#""\"" qwen35-json-char* "\"""#.to_string(),
    ));
    rules.push(("qwen35-json-colon".to_string(), r#"":" " "?"#.to_string()));
    rules.push(("qwen35-json-comma".to_string(), r#""," " "?"#.to_string()));
    rules.push((
        "qwen35-json-obj".to_string(),
        r#""{" ("}" | qwen35-json-str qwen35-json-colon qwen35-json-val (qwen35-json-comma qwen35-json-str qwen35-json-colon qwen35-json-val)* "}")"#.to_string(),
    ));
    rules.push((
        "qwen35-json-arr".to_string(),
        r#""[" ("]" | qwen35-json-val (qwen35-json-comma qwen35-json-val)* "]")"#.to_string(),
    ));
    rules.push((
        "qwen35-json-val".to_string(),
        "qwen35-json-str | qwen35-num-val | qwen35-bool-val | qwen35-null-val | qwen35-json-obj | qwen35-json-arr".to_string(),
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

    // Build the per-call body (shape-independent).  `single_body` is the
    // GBNF expression for exactly one Qwen 3.5/3.6 tool-call body
    // (`<function=NAME>...\n</function>`).  The shape selector below
    // wraps it in either the body-only (SingleBody) or marker-wrapped +
    // repeat (OneOrMoreCalls) shape.
    let single_body: String = if let Some(props) = properties {
        if props.is_empty() {
            format!("{} {}", func_open_lit, func_close_lit)
        } else {
            let mut required_block_names: Vec<String> = Vec::new();
            let mut optional_block_names: Vec<String> = Vec::new();
            let mut rule_counter: u32 = 0;
            let mut sorted_keys: Vec<&String> = props.keys().collect();
            sorted_keys.sort();
            for key in &sorted_keys {
                let val_schema = &props[*key];
                // iter-231b: array/object compile via the recursive
                // nested-schema converter (full declared-structure
                // fidelity; free-form shapes stay permissive).
                let val_rule = qwen35_value_rule(
                    fn_name,
                    key.as_str(),
                    val_schema,
                    &mut rules,
                    &mut rule_counter,
                )?;
                let param_open_lit = gbnf_literal(&format!("<parameter={}>", key));
                let param_close_lit = gbnf_literal("</parameter>");
                // Block form confirmed against tokenizer_config.json chat_template:
                //   `<parameter=KEY>\nVAL\n</parameter>\n`
                // Every </parameter> — including the last before </function> —
                // is followed by \n.  The template loop emits
                //   `{{- '\n</parameter>\n' }}` unconditionally.
                let block_body = format!(
                    "{} {} {} {} {} {}",
                    param_open_lit,
                    newline_lit,
                    val_rule,
                    newline_lit,
                    param_close_lit,
                    newline_lit
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
                rules.push((param_list_rule.clone(), format!("{}*", param_item_rule)));
                param_list_rule
            } else {
                // Case B: required permutation + optional suffix.
                let req_top =
                    build_qwen35_required_permutation(fn_name, &required_block_names, &mut rules);
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

            // Body: `<function=NAME>\n` + param sequence + `</function>`.
            // The newline after `<function=NAME>` is part of the template
            // emission pattern (verified from tokenizer_config.json).
            format!(
                "{} {} {} {}",
                func_open_lit, newline_lit, param_body_rule, func_close_lit
            )
        }
    } else {
        rules.push((
            "qwen35-inner-char".to_string(),
            // No-schema function bodies have the same delimiter ambiguity as
            // scalar parameters, but reserve `</function>` instead.
            r#"[^<\\] | [\\] [^\x00-\x1F] | "<" [^/] | "</" [^f] | "</f" [^u] | "</fu" [^n] | "</fun" [^c] | "</func" [^t] | "</funct" [^i] | "</functi" [^o] | "</functio" [^n] | "</function" [^>]"#
                .to_string(),
        ));
        format!("{} qwen35-inner-char* {}", func_open_lit, func_close_lit)
    };

    // Wave 2.7 W-η + Wave 3.5 HIGH-1: select root rule per `shape`.
    //
    //   * `SingleBody`                       → body-only root (single
    //     call, no markers; legacy lazy path).
    //   * `OneOrMoreCalls{p}`                → wrap body with Qwen
    //     3.5/3.6 marker pair `<tool_call>\n` / `\n</tool_call>`
    //     (verbatim from
    //     models/qwen3.6-27b-dwq46/tokenizer_config.json:285+) and
    //     repeat per `parallel`.  Calls 2+ are separated by a literal
    //     `\n` per the chat_template's `{%- else %}{{- '\n<tool_call>\n...
    //     ' }}` branch.
    //   * `OneOrMoreCallsBodyOnly{p}`        → Wave 3.5 HIGH-1 Auto-lazy
    //     variant.  Strips the FIRST open marker (consumed by the
    //     awaiting_trigger no-op gate before the grammar is engaged).
    //     Single: `body \n close_marker`; parallel:
    //     `body \n close_marker ( \n open_marker \n body \n close_marker )* space`.
    //     See `GrammarShape::OneOrMoreCallsBodyOnly` doc-comment for the
    //     production-order rationale.
    // iter-219b grammar-exhaust fix (2026-05-01): drop trailing ` space`
    // from all root_body shapes (sibling to the gemma4 emitter above; see
    // its block-comment for full rationale + Agent C audit citation).
    let root_body = match shape {
        GrammarShape::SingleBody => single_body.clone(),
        GrammarShape::OneOrMoreCalls { parallel } => {
            let open_marker = gbnf_literal("<tool_call>");
            let close_marker = gbnf_literal("</tool_call>");
            let separator_ws = "[ \\t\\r\\n]*";
            let qwen_call_rule = "qwen35-call".to_string();
            // Full call shape per chat_template:
            //   `<tool_call>\n<function=NAME>\n...\n</function>\n</tool_call>`
            // (single_body already starts with `<function=NAME>\n` and ends
            // with `</function>` — we add the surrounding `<tool_call>\n` and
            // `\n</tool_call>`).
            rules.push((
                qwen_call_rule.clone(),
                format!(
                    "{} {} {} {} {}",
                    open_marker, newline_lit, single_body, newline_lit, close_marker
                ),
            ));
            if parallel {
                // `call ("\n" call)*` — Qwen separates calls 2+ with literal `\n`
                // (template's loop "else" branch — see citation above).
                format!(
                    "{} {} ( {} {} )*",
                    separator_ws, qwen_call_rule, newline_lit, qwen_call_rule
                )
            } else {
                format!("{} {}", separator_ws, qwen_call_rule)
            }
        }
        GrammarShape::OneOrMoreCallsBodyOnly { parallel } => {
            // Wave 3.5 HIGH-1: leading open marker is stripped (consumed
            // by the awaiting_trigger gate before the grammar fires).
            // The trailing close marker IS in the grammar — those bytes
            // flow through `accept_bytes` once the runtime is eager.
            let open_marker = gbnf_literal("<tool_call>");
            let close_marker = gbnf_literal("</tool_call>");
            if parallel {
                // First call: `body \n close_marker` (no leading
                // `<tool_call>\n`).  Subsequent calls: full
                // `\n <tool_call> \n body \n </tool_call>` (matches the
                // chat_template loop's `\n<tool_call>\n...\n</tool_call>`
                // separator pattern).
                let qwen_call_rule = "qwen35-call".to_string();
                rules.push((
                    qwen_call_rule.clone(),
                    format!(
                        "{} {} {} {} {}",
                        open_marker, newline_lit, single_body, newline_lit, close_marker
                    ),
                ));
                format!(
                    "{} {} {} ( {} {} )*",
                    single_body, newline_lit, close_marker, newline_lit, qwen_call_rule
                )
            } else {
                // Single call: body + `\n</tool_call>` (close marker
                // preceded by `\n` per the template emission pattern).
                format!("{} {} {}", single_body, newline_lit, close_marker)
            }
        }
    };
    rules.push(("root".to_string(), root_body));

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

fn deepseek4_value_variants(
    fn_name: &str,
    path: &str,
    schema: &serde_json::Value,
    rules: &mut Vec<(String, String)>,
    rule_counter: &mut u32,
) -> Result<Vec<(bool, String)>, EmitterError> {
    if schema == &serde_json::Value::Bool(false) {
        return Ok(vec![(false, r#"[^\U00000000-\U0010FFFF]"#.to_string())]);
    }
    let Some(object) = schema.as_object() else {
        return Ok(vec![
            (true, "dsml-string-val".to_string()),
            (false, "qwen35-json-val".to_string()),
        ]);
    };
    for keyword in ["anyOf", "oneOf"] {
        if let Some(branches) = object.get(keyword).and_then(serde_json::Value::as_array) {
            let mut variants = Vec::new();
            for (index, branch) in branches.iter().enumerate() {
                variants.extend(deepseek4_value_variants(
                    fn_name,
                    &format!("{path}/{keyword}/{index}"),
                    branch,
                    rules,
                    rule_counter,
                )?);
            }
            return Ok(variants);
        }
    }
    if let Some(values) = object.get("enum").and_then(serde_json::Value::as_array) {
        let mut string_values = Vec::new();
        let mut json_values = Vec::new();
        for value in values {
            match value {
                serde_json::Value::String(value) => string_values.push(gbnf_literal(value)),
                serde_json::Value::Number(_)
                | serde_json::Value::Bool(_)
                | serde_json::Value::Null
                | serde_json::Value::Array(_)
                | serde_json::Value::Object(_) => {
                    let serialized = serde_json::to_string(value).map_err(|error| {
                        EmitterError::UnsupportedSchemaFeature {
                            fn_name: fn_name.to_string(),
                            param_path: path.to_string(),
                            feature: format!("enum serialize: {error}"),
                        }
                    })?;
                    json_values.push(gbnf_literal(&serialized));
                }
            }
        }
        let mut variants = Vec::new();
        if !string_values.is_empty() {
            variants.push((true, format!("( {} )", string_values.join(" | "))));
        }
        if !json_values.is_empty() {
            variants.push((false, format!("( {} )", json_values.join(" | "))));
        }
        return Ok(variants);
    }
    if object.get("type").and_then(serde_json::Value::as_str) == Some("string") {
        let value = match compile_string_assertion(
            fn_name,
            path,
            object,
            crate::serve::api::grammar::regex_gbnf::Surface::DeepSeekRawString,
        )? {
            Some(body) => body,
            None => bounded_string_body(object, "dsml-string-char")
                .unwrap_or_else(|| "dsml-string-val".to_string()),
        };
        return Ok(vec![(true, value)]);
    }
    Ok(vec![(
        false,
        qwen35_nested_value_rule(fn_name, path, schema, rules, rule_counter, 1)?,
    )])
}

/// Emit DeepSeek-V4's native DSML tool-call syntax. The outer marker pair
/// contains one or more invoke elements; parameter names are restricted to
/// the declared schema and non-string values are constrained to valid JSON.
fn deepseek4_tool_call_gbnf(
    fn_name: &str,
    params_schema: &serde_json::Value,
    shape: GrammarShape,
) -> Result<String, EmitterError> {
    let mut rules: Vec<(String, String)> = vec![
        (
            "qwen35-json-char".into(),
            r#"[^"\\\x00-\x1F] | [\\] (["\\/bfnrt] | "u" [0-9a-fA-F]{4})"#.into(),
        ),
        (
            "qwen35-json-str".into(),
            r#""\"" qwen35-json-char* "\"""#.into(),
        ),
        // The recursive structured-value compiler is shared with Qwen. DSML
        // examples commonly render one ASCII space after JSON separators,
        // while compact JSON is equally valid. Accept both surfaces so the
        // grammar does not mask the model's preferred `: "value"` token and
        // force a low-probability compact-JSON continuation.
        ("qwen35-json-colon".into(), r#"":" " "?"#.into()),
        ("qwen35-json-comma".into(), r#""," " "?"#.into()),
        (
            "qwen35-int-val".into(),
            r#""-"? ([0] | [1-9] [0-9]{0,15})"#.into(),
        ),
        (
            "qwen35-num-val".into(),
            r#""-"? ([0] | [1-9] [0-9]{0,15}) ("." [0-9]{1,16})? ([eE] [-+]? [0-9]{1,16})?"#.into(),
        ),
        ("qwen35-bool-val".into(), r#""true" | "false""#.into()),
        ("qwen35-null-val".into(), r#""null""#.into()),
        (
            "qwen35-json-obj".into(),
            r#""{" ("}" | qwen35-json-str qwen35-json-colon qwen35-json-val (qwen35-json-comma qwen35-json-str qwen35-json-colon qwen35-json-val)* "}")"#.into(),
        ),
        (
            "qwen35-json-arr".into(),
            r#""[" ("]" | qwen35-json-val (qwen35-json-comma qwen35-json-val)* "]")"#.into(),
        ),
        (
            "qwen35-json-val".into(),
            r#"qwen35-json-str | qwen35-num-val | qwen35-bool-val | qwen35-null-val | qwen35-json-obj | qwen35-json-arr"#.into(),
        ),
        (
            "dsml-string-char".into(),
            // String-valued DSML parameters carry raw tool arguments, so
            // source code and shell syntax must be able to contain `<`.
            // The following literal close-tag rule still delimits the value;
            // GrammarRuntime retains both branches while a `<` prefix is
            // ambiguous. Excluding every `<` forced coding calls such as
            // `fmt::Formatter<'_>` to close at `fmt::Formatter`.
            r#"[^\\] | [\\] [^\x00-\x1F]"#.into(),
        ),
        ("dsml-string-val".into(), "dsml-string-char*".into()),
    ];

    let newline = gbnf_literal("\n");
    let parameter_close = gbnf_literal("</｜DSML｜parameter>");
    let required_set: std::collections::HashSet<String> = params_schema
        .as_object()
        .and_then(|object| object.get("required"))
        .and_then(serde_json::Value::as_array)
        .map(|items| {
            items
                .iter()
                .filter_map(serde_json::Value::as_str)
                .map(ToOwned::to_owned)
                .collect()
        })
        .unwrap_or_default();
    if required_set.len() > MAX_REQUIRED_KEYS {
        return Err(EmitterError::TooManyRequiredKeys {
            fn_name: fn_name.to_string(),
            count: required_set.len(),
        });
    }

    let mut required_parameter_rules = Vec::new();
    let mut optional_parameter_rules = Vec::new();
    let mut rule_counter = 0_u32;
    if let Some(properties) = params_schema
        .as_object()
        .and_then(|object| object.get("properties"))
        .and_then(serde_json::Value::as_object)
    {
        let mut keys = properties.keys().collect::<Vec<_>>();
        keys.sort();
        for key in keys {
            let schema = &properties[key];
            let variants = deepseek4_value_variants(
                fn_name,
                &format!("/{key}"),
                schema,
                &mut rules,
                &mut rule_counter,
            )?;
            let rule_name = format!("dsml-param-{}", sanitize_rule_name_local(key));
            let alternatives = variants
                .into_iter()
                .map(|(is_string, value_rule)| {
                    let open = gbnf_literal(&format!(
                        "<｜DSML｜parameter name=\"{}\" string=\"{}\">",
                        key, is_string
                    ));
                    format!("{} {} {} {}", open, value_rule, parameter_close, newline)
                })
                .collect::<Vec<_>>();
            rules.push((
                rule_name.clone(),
                format!("( {} )", alternatives.join(" | ")),
            ));
            if required_set.contains(key.as_str()) {
                required_parameter_rules.push(rule_name);
            } else {
                optional_parameter_rules.push(rule_name);
            }
        }
    }
    let parameter_sequence =
        if required_parameter_rules.is_empty() && optional_parameter_rules.is_empty() {
            None
        } else if required_parameter_rules.is_empty() {
            let item = "dsml-optional-param".to_string();
            rules.push((
                item.clone(),
                format!("( {} )", optional_parameter_rules.join(" | ")),
            ));
            let list = "dsml-param-list".to_string();
            rules.push((list.clone(), format!("{}*", item)));
            Some(list)
        } else {
            let required =
                build_dsml_required_permutation(fn_name, &required_parameter_rules, &mut rules);
            if optional_parameter_rules.is_empty() {
                Some(required)
            } else {
                let item = "dsml-optional-param".to_string();
                rules.push((
                    item.clone(),
                    format!("( {} )", optional_parameter_rules.join(" | ")),
                ));
                let list = "dsml-param-list".to_string();
                rules.push((list.clone(), format!("{} {}*", required, item)));
                Some(list)
            }
        };

    let invoke_open = gbnf_literal(&format!("<｜DSML｜invoke name=\"{}\">", fn_name));
    let invoke_close = gbnf_literal("</｜DSML｜invoke>");
    let invoke = if let Some(parameters) = parameter_sequence {
        format!(
            "{} {} {} {}",
            invoke_open, newline, parameters, invoke_close
        )
    } else {
        // The official formatter emits one blank line for an empty argument
        // map: `<invoke>\n\n</invoke>`.
        format!("{} {} {} {}", invoke_open, newline, newline, invoke_close)
    };
    rules.push(("dsml-invoke".into(), invoke));

    let open = gbnf_literal("<｜DSML｜tool_calls>");
    let close = gbnf_literal("</｜DSML｜tool_calls>");
    let full_body = match shape {
        GrammarShape::SingleBody => "dsml-invoke".to_string(),
        GrammarShape::OneOrMoreCalls { parallel } => {
            let invokes = if parallel {
                format!("dsml-invoke ( {} dsml-invoke )*", newline)
            } else {
                "dsml-invoke".to_string()
            };
            format!("{} {} {} {} {}", open, newline, invokes, newline, close)
        }
        GrammarShape::OneOrMoreCallsBodyOnly { parallel } => {
            let invokes = if parallel {
                format!("dsml-invoke ( {} dsml-invoke )*", newline)
            } else {
                "dsml-invoke".to_string()
            };
            // The lazy splitter consumed only the outer open marker. The
            // leading newline, invoke body, and outer close remain constrained.
            format!("{} {} {} {}", newline, invokes, newline, close)
        }
    };
    rules.push(("root".into(), full_body));

    let mut output = String::new();
    output.push_str(&format!("root ::= {}\n", rules.last().unwrap().1));
    for (name, body) in rules.into_iter().filter(|(name, _)| name != "root") {
        output.push_str(&format!("{} ::= {}\n", name, body));
    }
    Ok(output)
}

/// Required DSML parameter blocks may appear in any order, but each must be
/// present exactly once. The recursive subset grammar matches the existing
/// Gemma/Qwen required-key contract and is bounded by `MAX_REQUIRED_KEYS`.
fn build_dsml_required_permutation(
    fn_name: &str,
    required: &[String],
    rules: &mut Vec<(String, String)>,
) -> String {
    let mut sorted = required.to_vec();
    sorted.sort();
    if sorted.len() == 1 {
        return sorted[0].clone();
    }
    let suffix = sorted
        .iter()
        .map(|name| name.trim_start_matches("dsml-param-"))
        .collect::<Vec<_>>()
        .join("-");
    let rule_name = format!(
        "dsml-required-{}-{}",
        sanitize_rule_name_local(fn_name),
        suffix
    );
    if rules.iter().any(|(name, _)| name == &rule_name) {
        return rule_name;
    }
    rules.push((rule_name.clone(), String::new()));
    let alternatives = sorted
        .iter()
        .enumerate()
        .map(|(index, item)| {
            let remainder = sorted
                .iter()
                .enumerate()
                .filter(|(candidate, _)| *candidate != index)
                .map(|(_, value)| value.clone())
                .collect::<Vec<_>>();
            let tail = build_dsml_required_permutation(fn_name, &remainder, rules);
            format!("{} {}", item, tail)
        })
        .collect::<Vec<_>>()
        .join(" | ");
    let (_, body) = rules
        .iter_mut()
        .find(|(name, _)| name == &rule_name)
        .expect("inserted DSML permutation rule");
    *body = alternatives;
    rule_name
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
/// iter-231a (B3 supersession): `array` and `object` types compile via the
/// iter-231b recursive nested-schema converter (`qwen35_nested_value_rule`).
fn qwen35_value_rule(
    fn_name: &str,
    param_name: &str,
    schema: &serde_json::Value,
    rules: &mut Vec<(String, String)>,
    rule_counter: &mut u32,
) -> Result<String, EmitterError> {
    if schema == &serde_json::Value::Bool(false) {
        return Ok(r#"[^\U00000000-\U0010FFFF]"#.to_string());
    }
    let obj = match schema.as_object() {
        Some(o) => o,
        None => return Ok(QWEN35_TOP_ANY_VAL.to_string()),
    };
    for keyword in ["anyOf", "oneOf"] {
        if let Some(serde_json::Value::Array(branches)) = obj.get(keyword) {
            let mut alternatives = Vec::with_capacity(branches.len());
            for (index, branch) in branches.iter().enumerate() {
                alternatives.push(qwen35_value_rule(
                    fn_name,
                    &format!("{param_name}-{keyword}-{index}"),
                    branch,
                    rules,
                    rule_counter,
                )?);
            }
            return Ok(format!("( {} )", alternatives.join(" | ")));
        }
    }
    // TOP-LEVEL strings are raw between parameter tags; every non-string is
    // rendered by the template's JSON path, including container values.
    if let Some(serde_json::Value::Array(values)) = obj.get("enum") {
        if values.is_empty() {
            return Ok(r#"[^\U00000000-\U0010FFFF]"#.to_string());
        }
        let alts = values
            .iter()
            .map(|value| match value {
                serde_json::Value::String(value) => Ok(gbnf_literal(value)),
                _ => serde_json::to_string(value)
                    .map(|value| gbnf_literal(&value))
                    .map_err(|error| EmitterError::UnsupportedSchemaFeature {
                        fn_name: fn_name.to_string(),
                        param_path: param_name.to_string(),
                        feature: format!("enum serialize: {error}"),
                    }),
            })
            .collect::<Result<Vec<_>, _>>()?;
        return Ok(format!("( {} )", alts.join(" | ")));
    }
    let schema_type = obj.get("type").and_then(|t| t.as_str()).unwrap_or("");
    match schema_type {
        "string" => {
            // iter-231c: `pattern` constrains the RAW string text.
            match compile_string_assertion(
                fn_name,
                &format!("/{}", param_name),
                obj,
                crate::serve::api::grammar::regex_gbnf::Surface::QwenRawString,
            )? {
                Some(body) => Ok(body),
                None => Ok(bounded_string_body(obj, "qwen35-str-char")
                    .unwrap_or_else(|| "qwen35-str-val".to_string())),
            }
        }
        "integer" => Ok(
            compile_integer_assertion(fn_name, &format!("/{}", param_name), obj)?
                .unwrap_or_else(|| "qwen35-int-val".to_string()),
        ),
        "number" => Ok("qwen35-num-val".to_string()),
        "boolean" => Ok("qwen35-bool-val".to_string()),
        "null" => Ok("qwen35-null-val".to_string()),
        // iter-231b: structured types → full-fidelity recursive compiler.
        "array" | "object" => qwen35_nested_value_rule(
            fn_name,
            &format!("/{}", param_name),
            schema,
            rules,
            rule_counter,
            1,
        ),
        _ => Ok(QWEN35_TOP_ANY_VAL.to_string()),
    }
}

/// Top-level value body for untyped / unknown-typed parameters: the
/// template renders string args RAW and non-string args via `tojson`, so
/// any scalar (raw) or any structured JSON value is accepted.
const QWEN35_TOP_ANY_VAL: &str = "( qwen35-any-val | qwen35-json-obj | qwen35-json-arr )";

/// iter-231c — compile a JSON Schema `pattern` keyword for the given
/// string surface via `grammar::regex_gbnf`.  Returns `Ok(None)` when
/// the schema carries no (string) `pattern`; `Err` for non-regular or
/// out-of-subset regex features (honest `UnsupportedSchemaFeature` —
/// never a silent permissive downgrade).
///
/// WHY the grammar-module dependency here: registry.rs previously had a
/// "no grammar module dependency" note (to avoid a circular import for a
/// trivial 10-line escape fn).  `regex_gbnf` has NO registry dependency,
/// so the dependency is one-directional and the original concern does
/// not apply.
fn compile_pattern(
    fn_name: &str,
    path: &str,
    pattern: Option<&serde_json::Value>,
    surface: crate::serve::api::grammar::regex_gbnf::Surface,
) -> Result<Option<String>, EmitterError> {
    let Some(pat) = pattern.and_then(|p| p.as_str()) else {
        return Ok(None);
    };
    match crate::serve::api::grammar::regex_gbnf::regex_to_gbnf_body(pat, surface) {
        Ok(body) => Ok(Some(body)),
        Err(e) => Err(EmitterError::UnsupportedSchemaFeature {
            fn_name: fn_name.to_string(),
            param_path: path.to_string(),
            feature: format!("pattern {:?}: {}", pat, e.0),
        }),
    }
}

fn compile_string_assertion(
    fn_name: &str,
    path: &str,
    object: &serde_json::Map<String, serde_json::Value>,
    surface: crate::serve::api::grammar::regex_gbnf::Surface,
) -> Result<Option<String>, EmitterError> {
    if let Some(format) = object.get("format").and_then(serde_json::Value::as_str) {
        return crate::serve::api::grammar::json_schema::string_format_gbnf(format, surface)
            .map(Some)
            .map_err(|error| EmitterError::UnsupportedSchemaFeature {
                fn_name: fn_name.to_string(),
                param_path: path.to_string(),
                feature: error.to_string(),
            });
    }
    compile_pattern(fn_name, path, object.get("pattern"), surface)
}

fn compile_integer_assertion(
    fn_name: &str,
    path: &str,
    object: &serde_json::Map<String, serde_json::Value>,
) -> Result<Option<String>, EmitterError> {
    let has_bound = ["minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum"]
        .iter()
        .any(|keyword| object.contains_key(*keyword));
    if !has_bound {
        return Ok(None);
    }
    crate::serve::api::grammar::json_schema::integer_range_gbnf(object)
        .map(Some)
        .map_err(|error| EmitterError::UnsupportedSchemaFeature {
            fn_name: fn_name.to_string(),
            param_path: path.to_string(),
            feature: error.to_string(),
        })
}

fn bounded_repeat(atom: &str, min: u64, max: Option<u64>) -> String {
    match max {
        Some(upper) if upper == min => format!("{atom}{{{min}}}"),
        Some(upper) => format!("{atom}{{{min},{upper}}}"),
        None if min == 0 => format!("{atom}*"),
        None if min == 1 => format!("{atom}+"),
        None => format!("{atom}{{{min},}}"),
    }
}

fn bounded_string_body(
    object: &serde_json::Map<String, serde_json::Value>,
    atom: &str,
) -> Option<String> {
    let min = object
        .get("minLength")
        .and_then(serde_json::Value::as_u64)
        .unwrap_or(0);
    let max = object.get("maxLength").and_then(serde_json::Value::as_u64);
    (min > 0 || max.is_some()).then(|| bounded_repeat(atom, min, max))
}

fn bounded_array_body(item: &str, comma: &str, min: u64, max: Option<u64>) -> String {
    if max == Some(0) {
        return r#""[" "]""#.to_string();
    }
    let comma_item = format!("( {comma} {item} )");
    let sequence = if min == 0 {
        match max {
            None => format!("( {item} {comma_item}* )?"),
            Some(upper) => format!("( {item} {comma_item}{{0,{}}} )?", upper - 1),
        }
    } else {
        let required_tail = min - 1;
        let suffix = match max {
            None if required_tail == 0 => format!(" {comma_item}*"),
            None => format!(" {comma_item}{{{required_tail}}} {comma_item}*"),
            Some(upper) if upper == min && required_tail == 0 => String::new(),
            Some(upper) if upper == min => format!(" {comma_item}{{{required_tail}}}"),
            Some(upper) if required_tail == 0 => {
                format!(" {comma_item}{{0,{}}}", upper - min)
            }
            Some(upper) => format!(
                " {comma_item}{{{required_tail}}} {comma_item}{{0,{}}}",
                upper - min
            ),
        };
        format!("{item}{suffix}")
    };
    format!(r#""[" {sequence} "]""#)
}

/// iter-231b — recursive nested-schema compiler for Qwen 3.5/3.6 tool
/// parameters (`tojson` JSON surface).
///
/// Constrains everything the schema DECLARES and stays open only where
/// the schema itself is open:
///
///   * scalars / enums / type-unions / anyOf / oneOf — exact bodies.
///   * `object` with `properties` — declared keys with per-key value
///     grammars; `required` (≤12) enforced any-order via permutation;
///     optional keys follow in a Kleene suffix (SAME contract as the
///     top-level parameter grammar: required keys in any order, then
///     optional keys in any order; optional duplicates are accepted the
///     same way top-level duplicates are).  `additionalProperties:false`
///     closes the key set; unset/true adds a wildcard kv tail whose exact
///     decoded-key complement cannot re-match declared keys; a schema-valued
///     `additionalProperties` constrains extra values.
///   * `object`/`array` with NO declared shape → the permissive
///     recursive `qwen35-json-obj` / `qwen35-json-arr` rules (any
///     well-formed compact JSON — the free-form MCP case).
///   * `array` with `items` → typed item grammar; absent → permissive.
///
/// `pattern` (regex) is COMPILED into the value grammar via
/// `grammar::regex_gbnf` (iter-231c); non-regular regex features
/// (backreferences, look-around, \p{...}, bounds > 2000) are rejected.
///
/// Rejected with `EmitterError::UnsupportedSchemaFeature` (→ HTTP 400
/// naming the feature + dot-path): `allOf`, `$ref`/`$defs`,
/// `not`/`if`/`then`/`else`, `dependentSchemas`, tuple-form `items`,
/// >32 properties per object, depth > 32.
/// Constraint keywords the grammar cannot enforce (`minLength`,
/// `minimum`, `minItems`, `format`, …) are IGNORED, mirroring
/// json_schema.rs's deferred-feature list.
///
fn qwen35_nested_value_rule(
    fn_name: &str,
    path: &str,
    schema: &serde_json::Value,
    rules: &mut Vec<(String, String)>,
    rule_counter: &mut u32,
    depth: usize,
) -> Result<String, EmitterError> {
    if depth > MAX_NESTED_DEPTH {
        return Err(EmitterError::UnsupportedSchemaFeature {
            fn_name: fn_name.to_string(),
            param_path: path.to_string(),
            feature: format!("nesting depth > {}", MAX_NESTED_DEPTH),
        });
    }
    if schema == &serde_json::Value::Bool(false) {
        return Ok(r#"[^\U00000000-\U0010FFFF]"#.to_string());
    }
    let obj = match schema.as_object() {
        Some(o) => o,
        // Non-object schema node (true/false/nonsense) → permissive leaf.
        None => return Ok("qwen35-json-val".to_string()),
    };

    validate_unconstrained_property_names(fn_name, path, obj.get("propertyNames"))?;

    // Hard-reject features the grammar cannot enforce (honest 400, no
    // silent downgrade).
    for feat in [
        "allOf",
        "$ref",
        "$defs",
        "not",
        "if",
        "then",
        "else",
        "dependentSchemas",
        "patternProperties",
        "contains",
    ] {
        if obj.contains_key(feat) {
            return Err(EmitterError::UnsupportedSchemaFeature {
                fn_name: fn_name.to_string(),
                param_path: path.to_string(),
                feature: feat.to_string(),
            });
        }
    }

    // anyOf / oneOf → alternation of sub-schema bodies (same acceptance
    // for grammar purposes — a value matching ANY branch is accepted).
    for comb in ["anyOf", "oneOf"] {
        if let Some(serde_json::Value::Array(subs)) = obj.get(comb) {
            if subs.is_empty() {
                return Err(EmitterError::UnsupportedSchemaFeature {
                    fn_name: fn_name.to_string(),
                    param_path: path.to_string(),
                    feature: format!("empty {}", comb),
                });
            }
            let mut alts: Vec<String> = Vec::with_capacity(subs.len());
            for (i, s) in subs.iter().enumerate() {
                alts.push(qwen35_nested_value_rule(
                    fn_name,
                    &format!("{}/{}/{}", path, comb, i),
                    s,
                    rules,
                    rule_counter,
                    depth + 1,
                )?);
            }
            return Ok(format!("( {} )", alts.join(" | ")));
        }
    }

    // enum → literal alternation of every JSON-serialized value (compact
    // tojson form; strings keep their JSON quotes).
    if let Some(serde_json::Value::Array(values)) = obj.get("enum") {
        if values.is_empty() {
            return Ok(r#"[^\U00000000-\U0010FFFF]"#.to_string());
        }
        let mut alts: Vec<String> = Vec::with_capacity(values.len());
        for v in values {
            let text =
                serde_json::to_string(v).map_err(|e| EmitterError::UnsupportedSchemaFeature {
                    fn_name: fn_name.to_string(),
                    param_path: path.to_string(),
                    feature: format!("enum serialize: {}", e),
                })?;
            alts.push(gbnf_literal(&text));
        }
        return Ok(format!("( {} )", alts.join(" | ")));
    }

    match obj.get("type") {
        // Untyped nested node → any JSON value (structured included).
        None => Ok("qwen35-json-val".to_string()),
        Some(serde_json::Value::Array(types)) => {
            let mut alts: Vec<String> = Vec::with_capacity(types.len());
            for (i, t) in types.iter().enumerate() {
                let Some(tstr) = t.as_str() else {
                    return Err(EmitterError::UnsupportedSchemaFeature {
                        fn_name: fn_name.to_string(),
                        param_path: path.to_string(),
                        feature: "non-string type union entry".to_string(),
                    });
                };
                let mut stub = serde_json::Map::new();
                stub.insert("type".into(), serde_json::Value::String(tstr.into()));
                alts.push(qwen35_nested_value_rule(
                    fn_name,
                    &format!("{}/type/{}", path, i),
                    &serde_json::Value::Object(stub),
                    rules,
                    rule_counter,
                    depth + 1,
                )?);
            }
            Ok(format!("( {} )", alts.join(" | ")))
        }
        Some(serde_json::Value::String(t)) => match t.as_str() {
            "string" => {
                // iter-231c: `pattern` constrains the JSON string content.
                match compile_string_assertion(
                    fn_name,
                    path,
                    obj,
                    crate::serve::api::grammar::regex_gbnf::Surface::QwenJsonString,
                )? {
                    Some(body) => Ok(format!(
                        "{} {} {}",
                        gbnf_literal("\""),
                        body,
                        gbnf_literal("\"")
                    )),
                    None => match bounded_string_body(obj, "qwen35-json-char") {
                        Some(body) => Ok(format!(
                            "{} {} {}",
                            gbnf_literal("\""),
                            body,
                            gbnf_literal("\"")
                        )),
                        None => Ok("qwen35-json-str".to_string()),
                    },
                }
            }
            "integer" => Ok(compile_integer_assertion(fn_name, path, obj)?
                .unwrap_or_else(|| "qwen35-int-val".to_string())),
            "number" => Ok("qwen35-num-val".to_string()),
            "boolean" => Ok("qwen35-bool-val".to_string()),
            "null" => Ok("qwen35-null-val".to_string()),
            "object" => qwen35_nested_object(fn_name, path, obj, rules, rule_counter, depth),
            "array" => qwen35_nested_array(fn_name, path, obj, rules, rule_counter, depth),
            other => Err(EmitterError::UnsupportedSchemaFeature {
                fn_name: fn_name.to_string(),
                param_path: path.to_string(),
                feature: format!("type '{}'", other),
            }),
        },
        Some(_) => Err(EmitterError::UnsupportedSchemaFeature {
            fn_name: fn_name.to_string(),
            param_path: path.to_string(),
            feature: "non-string type".to_string(),
        }),
    }
}

/// iter-231b — nested `object` compiler (Qwen compact-JSON surface).
/// See `qwen35_nested_value_rule` for the contract.
fn qwen35_nested_object(
    fn_name: &str,
    path: &str,
    obj: &serde_json::Map<String, serde_json::Value>,
    rules: &mut Vec<(String, String)>,
    rule_counter: &mut u32,
    depth: usize,
) -> Result<String, EmitterError> {
    let properties = obj.get("properties").and_then(|p| p.as_object());
    let additional_closed = matches!(
        obj.get("additionalProperties"),
        Some(serde_json::Value::Bool(false))
    );
    let additional_schema = obj.get("additionalProperties");

    let props = match properties {
        Some(p) if !p.is_empty() => p,
        _ => {
            if additional_closed {
                return Ok(r#""{" "}""#.to_string());
            }
            if additional_schema.is_some_and(serde_json::Value::is_object) {
                let value = qwen35_nested_value_rule(
                    fn_name,
                    &format!("{path}/additionalProperties"),
                    additional_schema.expect("present"),
                    rules,
                    rule_counter,
                    depth + 1,
                )?;
                return Ok(format!(
                    r#""{{" ( "}}" | qwen35-json-str qwen35-json-colon {value} ( qwen35-json-comma qwen35-json-str qwen35-json-colon {value} )* "}}" )"#
                ));
            }
            return Ok("qwen35-json-obj".to_string());
        }
    };

    let required_set: std::collections::HashSet<&str> = obj
        .get("required")
        .and_then(|v| v.as_array())
        .map(|arr| arr.iter().filter_map(|v| v.as_str()).collect())
        .unwrap_or_default();

    if required_set.len() > MAX_REQUIRED_KEYS {
        return Err(EmitterError::TooManyRequiredKeys {
            fn_name: format!("{} (nested {})", fn_name, path),
            count: required_set.len(),
        });
    }
    if props.len() > MAX_NESTED_PROPERTIES {
        return Err(EmitterError::UnsupportedSchemaFeature {
            fn_name: fn_name.to_string(),
            param_path: path.to_string(),
            feature: format!("> {} properties", MAX_NESTED_PROPERTIES),
        });
    }

    let mut req_kv: Vec<String> = Vec::new();
    let mut opt_kv: Vec<String> = Vec::new();
    let mut sorted_keys: Vec<&String> = props.keys().collect();
    sorted_keys.sort();
    for key in sorted_keys {
        let val_body = qwen35_nested_value_rule(
            fn_name,
            &format!("{}/properties/{}", path, key),
            &props[key],
            rules,
            rule_counter,
            depth + 1,
        )?;
        *rule_counter += 1;
        let kv_name = format!("q35n-{}-kv", *rule_counter);
        // Key literal: JSON-quoted exactly as serde emits it.
        let key_json = serde_json::to_string(key)
            .unwrap_or_else(|_| format!("\"{}\"", key.replace('"', "\\\"")));
        rules.push((
            kv_name.clone(),
            format!("{} qwen35-json-colon {}", gbnf_literal(&key_json), val_body),
        ));
        if required_set.contains(key.as_str()) {
            req_kv.push(kv_name);
        } else {
            opt_kv.push(kv_name);
        }
    }

    // Wildcard kv for open objects. The decoded-character trie excludes
    // declared names even when the model spells them with `\u00XX` escapes.
    let extra_kv: Option<String> = if additional_closed {
        None
    } else {
        let names = props.keys().cloned().collect::<Vec<_>>();
        let key_body = crate::serve::api::grammar::json_schema::json_string_excluding_gbnf(
            &names,
            "qwen35-json-char",
            true,
            true,
        )
        .map_err(|feature| EmitterError::UnsupportedSchemaFeature {
            fn_name: fn_name.to_string(),
            param_path: path.to_string(),
            feature,
        })?;
        let value_body = match additional_schema {
            Some(schema) => qwen35_nested_value_rule(
                fn_name,
                &format!("{path}/additionalProperties"),
                schema,
                rules,
                rule_counter,
                depth + 1,
            )?,
            None => "qwen35-json-val".to_string(),
        };
        *rule_counter += 1;
        let name = format!("q35n-{}-extra-kv", *rule_counter);
        rules.push((
            name.clone(),
            format!("{key_body} qwen35-json-colon {value_body}"),
        ));
        Some(name)
    };

    build_nested_obj_body(
        "q35n",
        req_kv,
        opt_kv,
        extra_kv,
        "qwen35-json-comma",
        rules,
        rule_counter,
    )
    .map(|inner| format!(r#""{{" {} "}}""#, inner))
}

/// iter-231b — nested `array` compiler (Qwen compact-JSON surface).
fn qwen35_nested_array(
    fn_name: &str,
    path: &str,
    obj: &serde_json::Map<String, serde_json::Value>,
    rules: &mut Vec<(String, String)>,
    rule_counter: &mut u32,
    depth: usize,
) -> Result<String, EmitterError> {
    let min = obj
        .get("minItems")
        .and_then(serde_json::Value::as_u64)
        .unwrap_or(0);
    let max = obj.get("maxItems").and_then(serde_json::Value::as_u64);
    match obj.get("items") {
        None if min == 0 && max.is_none() => Ok("qwen35-json-arr".to_string()),
        None => Ok(bounded_array_body(
            "qwen35-json-val",
            "qwen35-json-comma",
            min,
            max,
        )),
        Some(serde_json::Value::Object(_)) | Some(serde_json::Value::Bool(_)) => {
            let item_rule = qwen35_nested_value_rule(
                fn_name,
                &format!("{}/items", path),
                obj.get("items").expect("items checked above"),
                rules,
                rule_counter,
                depth + 1,
            )?;
            Ok(bounded_array_body(
                &item_rule,
                "qwen35-json-comma",
                min,
                max,
            ))
        }
        Some(serde_json::Value::Array(_)) => Err(EmitterError::UnsupportedSchemaFeature {
            fn_name: fn_name.to_string(),
            param_path: path.to_string(),
            feature: "tuple-form items".to_string(),
        }),
        Some(_) => Err(EmitterError::UnsupportedSchemaFeature {
            fn_name: fn_name.to_string(),
            param_path: path.to_string(),
            feature: "non-object items".to_string(),
        }),
    }
}

/// iter-231b — shared nested-object inner builder.  Emits the
/// required-permutation + optional-Kleene-suffix body used by BOTH
/// families' nested object compilers (same contract as the top-level
/// parameter grammars): required keys in any order first, then optional
/// keys (plus the wildcard tail for open objects) in any order.
///
/// Returns the GBNF expression for the object's inner kv sequence (no
/// surrounding braces).
fn build_nested_obj_body(
    prefix: &str,
    req_kv: Vec<String>,
    opt_kv: Vec<String>,
    extra_kv: Option<String>,
    comma: &str,
    rules: &mut Vec<(String, String)>,
    rule_counter: &mut u32,
) -> Result<String, EmitterError> {
    let mut opt_items: Vec<String> = opt_kv;
    if let Some(eb) = &extra_kv {
        opt_items.push(format!("( {} )", eb));
    }

    if req_kv.is_empty() {
        // All-optional object: `( item ("," item)* )?` — {} accepted.
        if opt_items.is_empty() {
            // Closed object, no declared keys → empty body; caller wraps
            // braces: `{" "}` shape handled by the caller passing an empty
            // optional set for a closed object with properties that all
            // got filtered (defensive; today unreachable).
            return Ok(String::new());
        }
        *rule_counter += 1;
        let item_name = format!("{}-{}-item", prefix, *rule_counter);
        rules.push((item_name.clone(), format!("( {} )", opt_items.join(" | "))));
        Ok(format!("( {} ( {} {} )* )?", item_name, comma, item_name))
    } else {
        let req_top = build_nested_kv_permutation(
            &format!("{}-{}", prefix, *rule_counter),
            &req_kv,
            comma,
            rules,
        );
        if opt_items.is_empty() {
            Ok(req_top)
        } else {
            *rule_counter += 1;
            let opt_name = format!("{}-{}-opt", prefix, *rule_counter);
            rules.push((opt_name.clone(), format!("( {} )", opt_items.join(" | "))));
            Ok(format!("{} ( {} {} )*", req_top, comma, opt_name))
        }
    }
}

/// iter-231b — shared any-order permutation builder for nested kv rules.
/// Mirrors `build_qwen35_required_permutation` but parameterized on the
/// rule-name prefix + separator so both families' nested compilers share
/// one implementation.  Sub-rule names are content-addressed (sorted
/// join), so shared subsets across branches are emitted once.
fn build_nested_kv_permutation(
    set_name: &str,
    item_names: &[String],
    separator_lit: &str,
    rules: &mut Vec<(String, String)>,
) -> String {
    let mut sorted = item_names.to_vec();
    sorted.sort();

    if sorted.len() == 1 {
        return sorted[0].clone();
    }

    let rule_name = format!("{}-{}", set_name, sorted.join("-"));
    if rules.iter().any(|(n, _)| n == &rule_name) {
        return rule_name;
    }

    let mut alts: Vec<String> = Vec::new();
    for (i, item) in sorted.iter().enumerate() {
        let remaining: Vec<String> = sorted
            .iter()
            .enumerate()
            .filter(|(j, _)| *j != i)
            .map(|(_, s)| s.clone())
            .collect();
        let rest = build_nested_kv_permutation(set_name, &remaining, separator_lit, rules);
        alts.push(format!("{} {} {}", item, separator_lit, rest));
    }
    rules.push((rule_name.clone(), alts.join(" | ")));
    rule_name
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

    // ---- ADR-005 iter-230 B — forced-open seeding (AC-B1/AC-B2) ----

    /// Suffix-rule truth table (Decision B1) for the served templates'
    /// generation-prompt tails.
    #[test]
    fn iter230_b1_prompt_seeds_reasoning_open_truth_table() {
        // qwen thinking-on: template seeds an OPEN think block.
        assert!(prompt_seeds_reasoning_open(
            "<|im_start|>assistant\n<think>\n",
            &QWEN35
        ));
        // qwen thinking-off: pre-closed suppressor → NOT seeded.
        assert!(!prompt_seeds_reasoning_open(
            "<|im_start|>assistant\n<think>\n\n</think>\n\n",
            &QWEN35
        ));
        // Gemma thinking-on: bare model turn — the model emits its own
        // open marker in the completion → NOT seeded.
        assert!(!prompt_seeds_reasoning_open("<|turn>model\n", &GEMMA4));
        // Gemma thinking-off: closed-channel seed ends with the CLOSE
        // marker → NOT seeded.
        assert!(!prompt_seeds_reasoning_open(
            "<|turn>model\n<|channel>thought\n<channel|>",
            &GEMMA4
        ));
        // No reasoning markers → never seeded.
        let no_markers = ModelRegistration {
            family: "none",
            id_substrings: &["x"],
            reasoning_open: None,
            reasoning_close: None,
            tool_open: None,
            tool_close: None,
            tool_preamble: None,
        };
        assert!(!prompt_seeds_reasoning_open("<think>\n", &no_markers));
    }

    /// Seeded-open: close marker split across fragment boundaries; the
    /// pre-close span routes to Reasoning, post-close to Content.
    #[test]
    fn iter230_b1_seeded_open_close_marker_across_fragments() {
        let mut sp = make_reasoning_splitter(&QWEN35, true).unwrap();
        let mut reasoning = String::new();
        let mut content = String::new();
        for frag in ["I think", " therefore</th", "ink>I am"] {
            for (slot, s) in sp.feed(frag) {
                match slot {
                    SplitSlot::Reasoning => reasoning.push_str(&s),
                    SplitSlot::Content => content.push_str(&s),
                }
            }
        }
        if let Some((slot, s)) = sp.finish() {
            match slot {
                SplitSlot::Reasoning => reasoning.push_str(&s),
                SplitSlot::Content => content.push_str(&s),
            }
        }
        assert_eq!(reasoning, "I think therefore");
        assert_eq!(content, "I am");
    }

    #[test]
    fn qwen_tool_open_implicitly_closes_reasoning_without_consuming_marker() {
        let mut splitter = make_reasoning_splitter(&QWEN35, true).unwrap();
        let mut reasoning = String::new();
        let mut content = String::new();
        for fragment in [
            "I should inspect it<tool_",
            "call><function=read_file>",
            "</function></tool_call>",
        ] {
            for (slot, text) in splitter.feed(fragment) {
                match slot {
                    SplitSlot::Reasoning => reasoning.push_str(&text),
                    SplitSlot::Content => content.push_str(&text),
                }
            }
        }
        if let Some((slot, text)) = splitter.finish() {
            match slot {
                SplitSlot::Reasoning => reasoning.push_str(&text),
                SplitSlot::Content => content.push_str(&text),
            }
        }

        assert_eq!(reasoning, "I should inspect it");
        assert_eq!(
            content, "<tool_call><function=read_file></function></tool_call>",
            "the implicit boundary must preserve native tool syntax for ToolCallSplitter"
        );
        assert!(!splitter.in_reasoning());
    }

    /// A Qwen reasoning close is followed by two protocol newlines before
    /// visible content. They may arrive in separate decoded fragments, but
    /// must never leak into the OpenAI `content` field.
    #[test]
    fn agentic_grammar_contract_qwen_reasoning_separator_is_not_content() {
        let mut sp = make_reasoning_splitter(&QWEN35, true).unwrap();
        let mut reasoning = String::new();
        let mut content = String::new();
        for fragment in ["inspect the result</thi", "nk>\n", "\nEXACT"] {
            for (slot, text) in sp.feed(fragment) {
                match slot {
                    SplitSlot::Reasoning => reasoning.push_str(&text),
                    SplitSlot::Content => content.push_str(&text),
                }
            }
        }
        if let Some((slot, text)) = sp.finish() {
            match slot {
                SplitSlot::Reasoning => reasoning.push_str(&text),
                SplitSlot::Content => content.push_str(&text),
            }
        }
        assert_eq!(reasoning, "inspect the result");
        assert_eq!(content, "EXACT");

        let (content, reasoning) =
            split_full_output_forced(&QWEN35, "reason</think> intentional", true);
        assert_eq!(reasoning.as_deref(), Some("reason"));
        assert_eq!(content, " intentional", "non-protocol whitespace is data");
    }

    /// Seeded-open with NO close marker: finish() drains the tail to
    /// Reasoning — nothing leaks into content.
    #[test]
    fn iter230_b1_seeded_open_no_close_finish_routes_to_reasoning() {
        let (content, reasoning) =
            split_full_output_forced(&QWEN35, "endless pondering with no close", true);
        assert_eq!(content, "");
        assert_eq!(
            reasoning.as_deref(),
            Some("endless pondering with no close")
        );
    }

    /// Unseeded (forced_open=false) behavior is byte-identical to the
    /// legacy constructor on marker-bearing text.
    #[test]
    fn iter230_b1_unseeded_byte_identical_to_legacy() {
        let text = "Sure! <think>step by step</think>The answer is 42.";
        let legacy = split_full_output(&QWEN35, text);
        let forced_false = split_full_output_forced(&QWEN35, text, false);
        assert_eq!(legacy, forced_false);
        assert_eq!(legacy.0, "Sure! The answer is 42.");
        assert_eq!(legacy.1.as_deref(), Some("step by step"));
    }

    /// Redundant model-emitted open marker while seeded open is
    /// reasoning TEXT (today's nested-open behavior), not swallowed.
    #[test]
    fn iter230_b1_redundant_open_while_seeded_is_reasoning_text() {
        let (content, reasoning) =
            split_full_output_forced(&QWEN35, "<think>abc</think>done", true);
        assert_eq!(reasoning.as_deref(), Some("<think>abc"));
        assert_eq!(content, "done");
    }

    /// AC-B2 — factory-only construction pin: no engine-path module
    /// constructs a ReasoningSplitter directly; everything routes
    /// through `make_reasoning_splitter` so the forced-open seed can't
    /// be silently dropped. (Pattern built at runtime so this pin's own
    /// source stays out of the scan.)
    #[test]
    fn iter230_b2_factory_only_construction() {
        let direct = format!("{}::{}", "ReasoningSplitter", "from_registration");
        let modules: [(&str, &str); 4] = [
            ("engine.rs", include_str!("engine.rs")),
            ("engine_qwen35.rs", include_str!("engine_qwen35.rs")),
            ("engine_qwen3vl.rs", include_str!("engine_qwen3vl.rs")),
            ("handlers.rs", include_str!("handlers.rs")),
        ];
        for (name, src) in modules {
            assert_eq!(
                src.matches(&direct).count(),
                0,
                "{name}: direct ReasoningSplitter construction found — \
                 use registry::make_reasoning_splitter(reg, forced_open) \
                 so the iter-230 forced-open seed is threaded"
            );
        }
    }

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
        assert!(QWEN35.matches("Qwen/Qwen3.8-27B"));
        assert!(QWEN35.matches("qwen38-27b-q4_k_m"));
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
        assert_ne!(GEMMA4.reasoning_open, QWEN35.reasoning_open);
        assert_ne!(GEMMA4.reasoning_close, QWEN35.reasoning_close);
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
        let out = split(&GEMMA4, "a<|channel>b<channel|>c<|channel>d<channel|>e");
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
                (
                    SplitSlot::Reasoning,
                    "thought\nlet me compute 73 * 47".into()
                ),
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

    /// ADR-005 Phase 4 iter C closure (2026-05-01): non-streaming
    /// surface contract. The handler calls `split_full_output` to
    /// extract `reasoning_text` from the raw decoded stream, THEN
    /// `extract_tool_calls_from_text` runs over the remaining
    /// post-reasoning content to extract tool calls. This test locks
    /// in that `split_full_output` correctly partitions a stream that
    /// contains BOTH reasoning markers AND tool-call markers — the
    /// reasoning span is the only thing extracted; tool-call markers
    /// remain in `content` for downstream tool-call splitter to
    /// consume. Companion to the engine.rs streaming-side
    /// `replay_routes_reasoning_then_tool_call_in_correct_order`
    /// test. Without this, a regression that incorrectly interleaved
    /// the two splitters in `split_full_output` would silently
    /// surface only at LIVE-test time.
    #[test]
    fn split_full_output_preserves_tool_call_markers_in_content() {
        // Qwen 3.5/3.6 reasoning span + tool-call span. The reasoning
        // splitter only knows about reasoning markers; tool-call
        // markers must pass through untouched into the `content`
        // return slot.
        let raw_stream =
            "<think>I should call get_weather</think>OK, calling now: \
             <tool_call>\n<function=get_weather><parameter=city>Paris</parameter></function>\n</tool_call>";
        let (content, reasoning) = split_full_output(&QWEN35, raw_stream);
        assert_eq!(
            reasoning.as_deref(),
            Some("I should call get_weather"),
            "iter C non-streaming contract: reasoning span must be \
             extracted verbatim, markers swallowed"
        );
        assert!(
            !content.contains("<think>") && !content.contains("</think>"),
            "iter C non-streaming contract: reasoning markers must NOT \
             leak into content slot; got content: {content:?}"
        );
        assert!(
            content.contains("<tool_call>") && content.contains("</tool_call>"),
            "iter C+B-2 composition: tool-call markers MUST be preserved \
             in the content slot for the downstream ToolCallSplitter to \
             consume; ReasoningSplitter must not touch them. got content: {content:?}"
        );
        assert!(
            content.contains("OK, calling now:"),
            "iter C non-streaming contract: post-reasoning natural-language \
             preamble must be preserved verbatim; got content: {content:?}"
        );
    }

    /// ADR-005 Phase 4 iter C closure: non-streaming surface symmetric
    /// contract — pure reasoning followed by EOS (no post-reasoning
    /// content). The handler must produce `message.reasoning_content`
    /// populated, `message.content` empty (which serializes to `""`
    /// per OpenAI compat — the assistant gave a non-empty turn that's
    /// just thinking). This locks in the symmetric streaming-side
    /// `replay_streaming_origin_pure_reasoning_no_content` test at
    /// the helper level.
    #[test]
    fn split_full_output_pure_reasoning_returns_empty_content() {
        let (content, reasoning) =
            split_full_output(&QWEN35, "<think>only thinking, no answer</think>");
        assert_eq!(
            reasoning.as_deref(),
            Some("only thinking, no answer"),
            "pure-reasoning input must produce a populated reasoning slot"
        );
        assert_eq!(
            content, "",
            "pure-reasoning input (no content after </think>) must produce \
             empty content slot; got: {content:?}"
        );
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
        let out = tcfeed(&GEMMA4, "pre <|tool_call>call:f{x:1}<tool_call|> post");
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

    /// iter-219c (2026-05-01) — when the model emits a registered in-call
    /// resync marker (`<|tool_response>`, `<|channel>`, `<|turn>` for
    /// gemma4) MID-tool-call-body, the splitter MUST abort the call by
    /// emitting a synthetic ToolCallClose (with the partial body as
    /// ToolCallText) instead of absorbing the marker bytes verbatim.
    /// This closes the iter-218 LIVE-bug class structurally at the
    /// splitter level — the validity-gate + content-fallback-scrub
    /// downstream layers (iter-219b) catch the SYMPTOM; iter-219c catches
    /// the CAUSE.
    ///
    /// Pre-iter-219c: this test FAILS — splitter emits
    /// `ToolCallText("call:get_current<|tool_response>call:get_current_weather{...}")`.
    /// Post-iter-219c: this test PASSES — splitter emits
    /// `ToolCallText("call:get_current")`, `ToolCallClose` (synthetic),
    /// then re-enters non-tool-call state for the rest.
    #[test]
    fn iter219c_in_call_tool_response_aborts_with_synthetic_close() {
        let out = tcfeed(
            &GEMMA4,
            "<|tool_call>call:get_current<|tool_response>call:get_current_weather{x:1}<tool_call|>",
        );
        let joined = tc_coalesce(&out);
        // The splitter MUST recognize <|tool_response> as a resync marker
        // and abort the malformed call. The synthetic ToolCallClose fires
        // when the resync is observed; subsequent text re-enters non-tool-call
        // routing (Content), where the second `<|tool_call>` reopens a NEW
        // call cleanly.
        assert!(
            matches!(joined.as_slice(),
                [
                    ToolCallEvent::ToolCallOpen,
                    ToolCallEvent::ToolCallText(t1),
                    ToolCallEvent::ToolCallClose,
                    // Optional Content between the abort and the next open
                    // (resync marker bytes are swallowed; whatever followed
                    // before `<|tool_call>` becomes Content).
                    ..
                ] if t1 == "call:get_current"
            ),
            "iter-219c: splitter MUST abort on in-call <|tool_response> with \
             ToolCallText(partial body) + synthetic ToolCallClose. Got: {joined:#?}"
        );
        // Stronger: no event in the sequence should contain `<|tool_response>`
        // as text — the marker bytes MUST be swallowed.
        for ev in &joined {
            match ev {
                ToolCallEvent::ToolCallText(t) | ToolCallEvent::Content(t) => {
                    assert!(
                        !t.contains("<|tool_response>") && !t.contains("<tool_response|>"),
                        "iter-219c: resync marker leaked into text event: {ev:#?}"
                    );
                }
                _ => {}
            }
        }
    }

    /// iter-219c sibling — `<|channel>` mid-call (Gemma 4 reasoning
    /// channel reopened inside a tool body) MUST also abort. NOTE: in
    /// production the `ReasoningSplitter` runs upstream of
    /// `ToolCallSplitter` and would have absorbed `<|channel>...<channel|>`
    /// before reaching here — this test exercises the splitter in
    /// isolation to verify it recognizes `<|channel>` as a resync marker
    /// (so the call aborts cleanly even if upstream routing fails). The
    /// test does NOT assert the `<channel|>` close-marker is scrubbed
    /// from Content because that's `ReasoningSplitter`'s contract, not
    /// `ToolCallSplitter`'s.
    #[test]
    fn iter219c_in_call_channel_marker_aborts() {
        let out = tcfeed(
            &GEMMA4,
            "<|tool_call>call:f<|channel>thought<channel|>{x:1}<tool_call|>",
        );
        let joined = tc_coalesce(&out);
        // Iterate looking for the abort pattern: ToolCallOpen → ToolCallText
        // (containing only `call:f`) → ToolCallClose.
        let saw_clean_partial = joined.windows(3).any(|w| {
            matches!(w,
                [
                    ToolCallEvent::ToolCallOpen,
                    ToolCallEvent::ToolCallText(t),
                    ToolCallEvent::ToolCallClose,
                ] if t == "call:f"
            )
        });
        assert!(
            saw_clean_partial,
            "iter-219c: <|channel> mid-call MUST trigger abort; expected \
             ToolCallText(\"call:f\") + synthetic ToolCallClose; got: {joined:#?}"
        );
        // `<|channel>` (open marker, in resync set) MUST be swallowed.
        // `<channel|>` (close marker, NOT in our resync set; handled by
        // ReasoningSplitter upstream) may appear in Content here — that's
        // fine for the unit-isolation test; production never sees it.
        for ev in &joined {
            if let ToolCallEvent::ToolCallText(t) = ev {
                assert!(
                    !t.contains("<|channel>"),
                    "iter-219c: <|channel> leaked into ToolCallText: {ev:#?}"
                );
            }
        }
    }

    /// iter-219c regression-guard — close_marker that comes BEFORE any
    /// resync marker still wins (existing happy-path behavior unchanged).
    #[test]
    fn iter219c_close_before_resync_still_wins() {
        let out = tcfeed(
            &GEMMA4,
            // Clean call, then post-close `<|tool_response>` (legitimate —
            // host emits tool_response after the call closes; that's
            // post-close territory, not in-call).
            "<|tool_call>call:f{x:1}<tool_call|><|tool_response>tool_result<tool_response|>",
        );
        let joined = tc_coalesce(&out);
        // The first 4 events must be the clean happy-path:
        //   Open, Text("call:f{x:1}"), Close, Content(...)
        assert!(
            joined.len() >= 4,
            "iter-219c happy path: expected ≥4 events; got {joined:#?}"
        );
        match (&joined[0], &joined[1], &joined[2]) {
            (
                ToolCallEvent::ToolCallOpen,
                ToolCallEvent::ToolCallText(t),
                ToolCallEvent::ToolCallClose,
            ) if t == "call:f{x:1}" => {}
            _ => panic!("iter-219c: close_marker MUST win over later resync; got: {joined:#?}"),
        }
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
        let v: serde_json::Value = serde_json::from_str(&parsed.arguments_json).expect("arg JSON");
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
        let v: serde_json::Value = serde_json::from_str(&parsed.arguments_json).expect("arg JSON");
        assert_eq!(v["location"], "San Francisco");
        assert_eq!(v["unit"], "celsius");
    }

    #[test]
    fn parse_gemma4_numeric_and_bool_args() {
        let parsed = parse_tool_call_body(&GEMMA4, "call:f{count:42,enabled:true,ratio:1.5}")
            .expect("parse");
        let v: serde_json::Value = serde_json::from_str(&parsed.arguments_json).expect("arg JSON");
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
        let v: serde_json::Value = serde_json::from_str(&parsed.arguments_json).expect("arg JSON");
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
        let v: serde_json::Value = serde_json::from_str(&parsed.arguments_json).expect("arg JSON");
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
        let v: serde_json::Value = serde_json::from_str(&parsed.arguments_json).expect("arg JSON");
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
        let g = crate::serve::api::grammar::parser::parse_generated(gbnf)
            .unwrap_or_else(|e| panic!("parse GBNF:\n{}\nerror: {}", gbnf, e));
        let rid = g
            .rule_id("root")
            .unwrap_or_else(|| panic!("no root rule in GBNF:\n{}", gbnf));
        crate::serve::api::grammar::sampler::GrammarRuntime::new(g, rid)
            .unwrap_or_else(|| panic!("GrammarRuntime::new returned None for GBNF:\n{}", gbnf))
    }

    fn gemma4_runtime(
        fn_name: &str,
        schema_json: &str,
    ) -> crate::serve::api::grammar::sampler::GrammarRuntime {
        // Wave 2.7 W-η: legacy tests use body-only grammars (no markers in the
        // grammar; lazy/auto path).  Tests that exercise the eager
        // marker-wrapped form pass `OneOrMoreCalls{...}` directly via the
        // public emitter API.
        let schema: serde_json::Value = serde_json::from_str(schema_json).unwrap();
        let gbnf = GEMMA4
            .tool_call_gbnf(fn_name, &schema, GrammarShape::SingleBody)
            .unwrap_or_else(|e| panic!("tool_call_gbnf error: {}", e));
        grammar_runtime_for_gbnf(&gbnf)
    }

    fn qwen35_runtime(
        fn_name: &str,
        schema_json: &str,
    ) -> crate::serve::api::grammar::sampler::GrammarRuntime {
        let schema: serde_json::Value = serde_json::from_str(schema_json).unwrap();
        let gbnf = QWEN35
            .tool_call_gbnf(fn_name, &schema, GrammarShape::SingleBody)
            .unwrap_or_else(|e| panic!("tool_call_gbnf error: {}", e));
        grammar_runtime_for_gbnf(&gbnf)
    }

    fn deepseek4_runtime(
        fn_name: &str,
        schema_json: &str,
        shape: GrammarShape,
    ) -> crate::serve::api::grammar::sampler::GrammarRuntime {
        let schema: serde_json::Value = serde_json::from_str(schema_json).unwrap();
        let gbnf = DEEPSEEK4
            .tool_call_gbnf(fn_name, &schema, shape)
            .unwrap_or_else(|error| panic!("tool_call_gbnf error: {error}"));
        grammar_runtime_for_gbnf(&gbnf)
    }

    fn gemma4_fixture_value(value: &serde_json::Value) -> String {
        match value {
            serde_json::Value::Null => "null".to_string(),
            serde_json::Value::Bool(value) => value.to_string(),
            serde_json::Value::Number(value) => value.to_string(),
            serde_json::Value::String(value) => format!(r#"<|"|>{value}<|"|>"#),
            serde_json::Value::Array(values) => format!(
                "[{}]",
                values
                    .iter()
                    .map(gemma4_fixture_value)
                    .collect::<Vec<_>>()
                    .join(",")
            ),
            serde_json::Value::Object(values) => format!(
                "{{{}}}",
                values
                    .iter()
                    .map(|(key, value)| format!("{key}:{}", gemma4_fixture_value(value)))
                    .collect::<Vec<_>>()
                    .join(",")
            ),
        }
    }

    fn fixture_tool_wire(
        registration: &ModelRegistration,
        fn_name: &str,
        payload: &serde_json::Value,
    ) -> String {
        let object = payload.as_object().expect("fixture payload object");
        match registration.family {
            "gemma4" => format!(
                "call:{fn_name}{{{}}}",
                object
                    .iter()
                    .map(|(key, value)| format!("{key}:{}", gemma4_fixture_value(value)))
                    .collect::<Vec<_>>()
                    .join(",")
            ),
            "qwen35" => {
                let mut output = format!("<function={fn_name}>\n");
                for (key, value) in object {
                    let body = value.as_str().map(ToOwned::to_owned).unwrap_or_else(|| {
                        serde_json::to_string(value).expect("serialize fixture")
                    });
                    output.push_str(&format!("<parameter={key}>\n{body}\n</parameter>\n"));
                }
                output.push_str("</function>");
                output
            }
            "deepseek4" => {
                let mut output = format!("<｜DSML｜invoke name=\"{fn_name}\">\n");
                for (key, value) in object {
                    let (is_string, body) = match value.as_str() {
                        Some(value) => (true, value.to_string()),
                        None => (
                            false,
                            serde_json::to_string(value).expect("serialize fixture"),
                        ),
                    };
                    output.push_str(&format!(
                        "<｜DSML｜parameter name=\"{key}\" string=\"{is_string}\">{body}</｜DSML｜parameter>\n"
                    ));
                }
                output.push_str("</｜DSML｜invoke>");
                output
            }
            family => panic!("unexpected fixture family {family}"),
        }
    }

    fn assert_r2c_fixture_across_families(
        fn_name: &str,
        schema_json: &str,
        accepted_json: &[&str],
        rejected_json: &[&str],
    ) {
        let schema: serde_json::Value = serde_json::from_str(schema_json).expect("schema fixture");
        for registration in [&GEMMA4, &QWEN35, &DEEPSEEK4] {
            let grammar = registration
                .tool_call_gbnf(fn_name, &schema, GrammarShape::SingleBody)
                .unwrap_or_else(|error| panic!("{} schema compile: {error}", registration.family));
            for payload in accepted_json {
                let payload: serde_json::Value =
                    serde_json::from_str(payload).expect("valid payload fixture");
                let wire = fixture_tool_wire(registration, fn_name, &payload);
                let mut runtime = grammar_runtime_for_gbnf(&grammar);
                assert!(
                    runtime.accept_bytes(wire.as_bytes()) && runtime.is_accepted(),
                    "{} rejected valid {fn_name} wire:\n{wire}",
                    registration.family
                );
            }
            for payload in rejected_json {
                let payload: serde_json::Value =
                    serde_json::from_str(payload).expect("invalid payload fixture");
                let wire = fixture_tool_wire(registration, fn_name, &payload);
                let mut runtime = grammar_runtime_for_gbnf(&grammar);
                assert!(
                    !runtime.accept_bytes(wire.as_bytes()) || !runtime.is_accepted(),
                    "{} accepted invalid {fn_name} wire:\n{wire}",
                    registration.family
                );
            }
        }
    }

    #[test]
    fn r2c_stage6_schema_is_enforced_across_all_generative_families() {
        assert_r2c_fixture_across_families(
            "r2c_stage6",
            include_str!("../../../tests/fixtures/structured_output/r2c/stage6_review_lens.schema.json"),
            &[include_str!("../../../tests/fixtures/structured_output/r2c/stage6_review_lens.valid.json")],
            &[
                include_str!("../../../tests/fixtures/structured_output/r2c/stage6_review_lens.invalid_disposition.json"),
                include_str!("../../../tests/fixtures/structured_output/r2c/stage6_review_lens.invalid_missing_detail.json"),
            ],
        );
    }

    #[test]
    fn r2c_stage9_schema_is_enforced_across_all_generative_families() {
        assert_r2c_fixture_across_families(
            "r2c_stage9",
            include_str!("../../../tests/fixtures/structured_output/r2c/stage9_cwe.schema.json"),
            &[
                include_str!("../../../tests/fixtures/structured_output/r2c/stage9_cwe.valid.json"),
                include_str!("../../../tests/fixtures/structured_output/r2c/stage9_cwe.valid_abstention.json"),
            ],
            &[
                include_str!("../../../tests/fixtures/structured_output/r2c/stage9_cwe.invalid_cwe.json"),
                include_str!("../../../tests/fixtures/structured_output/r2c/stage9_cwe.invalid_abstention.json"),
            ],
        );
    }

    #[test]
    fn standard_format_and_integer_bounds_are_enforced_across_all_generative_families() {
        assert_r2c_fixture_across_families(
            "bounded_event",
            r#"{
                "type":"object",
                "properties":{
                    "date":{"type":"string","format":"date"},
                    "count":{"type":"integer","minimum":-2,"exclusiveMaximum":3}
                },
                "required":["date","count"],
                "additionalProperties":false
            }"#,
            &[r#"{"date":"2026-09-03","count":2}"#],
            &[
                r#"{"date":"2026-19-03","count":2}"#,
                r#"{"date":"2026-09-03","count":3}"#,
                r#"{"date":"2026-09-03","count":-3}"#,
            ],
        );
    }

    #[test]
    fn finite_composition_and_open_object_narrowing_hold_across_all_families() {
        assert_r2c_fixture_across_families(
            "narrowed_values",
            r#"{
                "type":"object",
                "properties":{
                    "count":{"type":"integer","enum":[1,2]},
                    "mode":{
                        "type":"string",
                        "minLength":2,
                        "enum":["x","ok"],
                        "anyOf":[{"const":"x"},{"const":"ok"}]
                    },
                    "exclusive":{
                        "type":"string",
                        "minLength":2,
                        "oneOf":[{"const":"x"},{"const":"ok"}]
                    },
                    "choice":{"enum":[{"ok":true},[1,2]]},
                    "cfg":{
                        "type":"object",
                        "properties":{"limit":{"type":"integer"}},
                        "additionalProperties":{"type":"string"}
                    }
                },
                "required":["count","mode","exclusive","choice","cfg"],
                "additionalProperties":false
            }"#,
            &[
                r#"{"count":1,"mode":"ok","exclusive":"ok","choice":{"ok":true},"cfg":{"limit":2,"note":"yes"}}"#,
            ],
            &[
                r#"{"count":3,"mode":"ok","exclusive":"ok","choice":{"ok":true},"cfg":{}}"#,
                r#"{"count":1,"mode":"x","exclusive":"ok","choice":{"ok":true},"cfg":{}}"#,
                r#"{"count":1,"mode":"ok","exclusive":"x","choice":{"ok":true},"cfg":{}}"#,
                r#"{"count":1,"mode":"ok","exclusive":"ok","choice":{"ok":false},"cfg":{}}"#,
                r#"{"count":1,"mode":"ok","exclusive":"ok","choice":{"ok":true},"cfg":{"limit":2.5}}"#,
                r#"{"count":1,"mode":"ok","exclusive":"ok","choice":{"ok":true},"cfg":{"note":3}}"#,
            ],
        );
    }

    #[test]
    fn native_tool_wires_reject_property_count_and_tuple_assertions_instead_of_widening() {
        for registration in [&GEMMA4, &QWEN35, &DEEPSEEK4] {
            for schema in [
                serde_json::json!({"type":"object","minProperties":1}),
                serde_json::json!({"type":"array","prefixItems":[{"type":"string"}]}),
            ] {
                let error = registration
                    .tool_call_gbnf("f", &schema, GrammarShape::SingleBody)
                    .expect_err("unsupported native-wire assertion must fail closed");
                assert!(
                    error.contains("cannot enforce JSON Schema assertion"),
                    "{}: {error}",
                    registration.family
                );
            }
        }
    }

    #[test]
    fn deepseek4_registration_and_multi_invoke_parser_are_openai_compatible() {
        let registration = find_for("DeepSeek-V4-Flash-0731").expect("DeepSeek registration");
        assert_eq!(registration.family, "deepseek4");
        assert_eq!(
            find_for("Deepseek v4 Flash 0731 Source")
                .expect("converted general.name registration")
                .family,
            "deepseek4"
        );
        let body = "\n<｜DSML｜invoke name=\"read_file\">\n<｜DSML｜parameter name=\"path\" string=\"true\">src/main.rs</｜DSML｜parameter>\n</｜DSML｜invoke>\n<｜DSML｜invoke name=\"run_tests\">\n<｜DSML｜parameter name=\"all\" string=\"false\">true</｜DSML｜parameter>\n</｜DSML｜invoke>\n";
        let calls = parse_tool_call_bodies(&registration, body).expect("parse DSML block");
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0].name, "read_file");
        assert_eq!(
            serde_json::from_str::<serde_json::Value>(&calls[0].arguments_json).unwrap(),
            serde_json::json!({"path": "src/main.rs"})
        );
        assert_eq!(calls[1].name, "run_tests");
        assert_eq!(
            serde_json::from_str::<serde_json::Value>(&calls[1].arguments_json).unwrap(),
            serde_json::json!({"all": true})
        );
    }

    #[test]
    fn deepseek4_required_grammar_enforces_required_parameter() {
        let schema = r#"{
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "line": {"type": "integer"}
            },
            "required": ["path"]
        }"#;
        let mut accepted = deepseek4_runtime(
            "read_file",
            schema,
            GrammarShape::OneOrMoreCalls { parallel: false },
        );
        let valid = "<｜DSML｜tool_calls>\n<｜DSML｜invoke name=\"read_file\">\n<｜DSML｜parameter name=\"path\" string=\"true\">src/lib.rs</｜DSML｜parameter>\n</｜DSML｜invoke>\n</｜DSML｜tool_calls>";
        assert!(accepted.accept_bytes(valid.as_bytes()));
        assert!(accepted.is_accepted());

        let mut rejected = deepseek4_runtime(
            "read_file",
            schema,
            GrammarShape::OneOrMoreCalls { parallel: false },
        );
        let missing = "<｜DSML｜tool_calls>\n<｜DSML｜invoke name=\"read_file\">\n<｜DSML｜parameter name=\"line\" string=\"false\">7</｜DSML｜parameter>\n</｜DSML｜invoke>\n</｜DSML｜tool_calls>";
        let alive = rejected.accept_bytes(missing.as_bytes());
        assert!(!(alive && rejected.is_accepted()));
    }

    #[test]
    fn deepseek4_question_grammar_rejects_observed_malformed_json() {
        let schema = r#"{
            "type": "object",
            "properties": {"questions": {"type": "array"}},
            "required": ["questions"]
        }"#;
        let mut runtime = deepseek4_runtime(
            "question",
            schema,
            GrammarShape::OneOrMoreCallsBodyOnly { parallel: false },
        );
        let malformed = "\n<｜DSML｜invoke name=\"question\">\n<｜DSML｜parameter name=\"questions\" string=\"false\">[{\"header\":null,\"options\":{\"label\":\"Movies\",}</｜DSML｜parameter>\n</｜DSML｜invoke>\n</｜DSML｜tool_calls>";
        let alive = runtime.accept_bytes(malformed.as_bytes());
        assert!(
            !(alive && runtime.is_accepted()),
            "malformed JSON must be impossible after the lazy tool boundary"
        );
    }

    #[test]
    fn deepseek4_question_nested_schema_rejects_null_header_and_accepts_repair() {
        let schema = r#"{
            "type": "object",
            "properties": {
                "questions": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "header": {"type": "string"},
                            "question": {"type": "string"},
                            "options": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "label": {"type": "string"},
                                        "description": {"type": "string"}
                                    },
                                    "required": ["label", "description"],
                                    "additionalProperties": false
                                }
                            },
                            "multiple": {"type": "boolean"}
                        },
                        "required": ["header", "question", "options"],
                        "additionalProperties": false
                    }
                }
            },
            "required": ["questions"],
            "additionalProperties": false
        }"#;
        let malformed = "\n<｜DSML｜invoke name=\"question\">\n<｜DSML｜parameter name=\"questions\" string=\"false\">[{\"header\":null,\"question\":\"What kind of video?\",\"options\":[{\"label\":\"Movies\",\"description\":\"Narrative film\"}]}]</｜DSML｜parameter>\n</｜DSML｜invoke>\n</｜DSML｜tool_calls>";
        let repaired = "\n<｜DSML｜invoke name=\"question\">\n<｜DSML｜parameter name=\"questions\" string=\"false\">[{\"header\":\"Video type\",\"question\":\"What kind of video?\",\"options\":[{\"label\":\"Movies\",\"description\":\"Narrative film\"}]}]</｜DSML｜parameter>\n</｜DSML｜invoke>\n</｜DSML｜tool_calls>";
        let repaired_spaced = "\n<｜DSML｜invoke name=\"question\">\n<｜DSML｜parameter name=\"questions\" string=\"false\">[{\"header\": \"Video type\", \"question\": \"What kind of video?\", \"options\": [{\"label\": \"Movies\", \"description\": \"Narrative film\"}]}]</｜DSML｜parameter>\n</｜DSML｜invoke>\n</｜DSML｜tool_calls>";

        let mut rejected = deepseek4_runtime(
            "question",
            schema,
            GrammarShape::OneOrMoreCallsBodyOnly { parallel: false },
        );
        assert!(
            !rejected.accept_bytes(malformed.as_bytes()),
            "a schema-required string header must not admit JSON null"
        );

        let mut accepted = deepseek4_runtime(
            "question",
            schema,
            GrammarShape::OneOrMoreCallsBodyOnly { parallel: false },
        );
        assert!(accepted.accept_bytes(repaired.as_bytes()));
        assert!(accepted.is_accepted());

        let mut accepted_spaced = deepseek4_runtime(
            "question",
            schema,
            GrammarShape::OneOrMoreCallsBodyOnly { parallel: false },
        );
        assert!(
            accepted_spaced.accept_bytes(repaired_spaced.as_bytes()),
            "DSML nested JSON must accept the model's canonical spaced separators"
        );
        assert!(accepted_spaced.is_accepted());

        let mut boundary = deepseek4_runtime(
            "question",
            schema,
            GrammarShape::OneOrMoreCallsBodyOnly { parallel: false },
        );
        let through_header_colon = "\n<｜DSML｜invoke name=\"question\">\n<｜DSML｜parameter name=\"questions\" string=\"false\">[{\"header\":";
        assert!(boundary.accept_bytes(through_header_colon.as_bytes()));
        assert!(
            boundary.accept_bytes(b" \""),
            "a token carrying space plus the opening string quote must survive"
        );
    }

    #[test]
    fn deepseek4_todowrite_nested_schema_rejects_null_content_and_accepts_repair() {
        // Stock OpenCode's TodoWrite shape: the outer `todos` array and every
        // item's content/status/priority fields are required strings.  This
        // regression is deliberately tool-name agnostic at the compiler
        // boundary; `todowrite` is the production witness that nested array
        // item schemas remain authoritative for DSML values.
        let schema = r#"{
            "type": "object",
            "properties": {
                "todos": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "content": {"type": "string"},
                            "status": {"type": "string"},
                            "priority": {"type": "string"}
                        },
                        "required": ["content", "status", "priority"],
                        "additionalProperties": false
                    }
                }
            },
            "required": ["todos"],
            "additionalProperties": false
        }"#;
        let malformed = "\n<｜DSML｜invoke name=\"todowrite\">\n<｜DSML｜parameter name=\"todos\" string=\"false\">[{\"content\":null,\"status\":\"in_progress\",\"priority\":\"high\"}]</｜DSML｜parameter>\n</｜DSML｜invoke>\n</｜DSML｜tool_calls>";
        let repaired = "\n<｜DSML｜invoke name=\"todowrite\">\n<｜DSML｜parameter name=\"todos\" string=\"false\">[{\"content\":\"Inspect the environment\",\"status\":\"in_progress\",\"priority\":\"high\"}]</｜DSML｜parameter>\n</｜DSML｜invoke>\n</｜DSML｜tool_calls>";

        let mut rejected = deepseek4_runtime(
            "todowrite",
            schema,
            GrammarShape::OneOrMoreCallsBodyOnly { parallel: false },
        );
        assert!(
            !rejected.accept_bytes(malformed.as_bytes()),
            "a schema-required todo content string must not admit JSON null"
        );

        let mut accepted = deepseek4_runtime(
            "todowrite",
            schema,
            GrammarShape::OneOrMoreCallsBodyOnly { parallel: false },
        );
        assert!(accepted.accept_bytes(repaired.as_bytes()));
        assert!(accepted.is_accepted());
    }

    #[test]
    fn agentic_grammar_contract_cross_family_question_rejects_null_header() {
        let schema = r#"{
            "type": "object",
            "properties": {
                "questions": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "header": {"type": "string"},
                            "question": {"type": "string"},
                            "options": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "label": {"type": "string"},
                                        "description": {"type": "string"}
                                    },
                                    "required": ["label", "description"],
                                    "additionalProperties": false
                                }
                            }
                        },
                        "required": ["header", "question", "options"],
                        "additionalProperties": false
                    }
                }
            },
            "required": ["questions"],
            "additionalProperties": false
        }"#;

        let qwen_bad = b"<function=question>\n<parameter=questions>\n[{\"header\":null,\"question\":\"What kind of video?\",\"options\":[{\"label\":\"Movies\",\"description\":\"Narrative film\"}]}]\n</parameter>\n</function>";
        let qwen_good = b"<function=question>\n<parameter=questions>\n[{\"header\":\"Video type\",\"question\":\"What kind of video?\",\"options\":[{\"label\":\"Movies\",\"description\":\"Narrative film\"}]}]\n</parameter>\n</function>";
        let mut qwen_rejected = qwen35_runtime("question", schema);
        assert!(!qwen_rejected.accept_bytes(qwen_bad));
        let mut qwen_accepted = qwen35_runtime("question", schema);
        assert!(qwen_accepted.accept_bytes(qwen_good));
        assert!(qwen_accepted.is_accepted());

        let gemma_bad = b"call:question{questions:[{header:null,question:<|\"|>What kind of video?<|\"|>,options:[{label:<|\"|>Movies<|\"|>,description:<|\"|>Narrative film<|\"|>}]}]}";
        let gemma_good = b"call:question{questions:[{header:<|\"|>Video type<|\"|>,question:<|\"|>What kind of video?<|\"|>,options:[{label:<|\"|>Movies<|\"|>,description:<|\"|>Narrative film<|\"|>}]}]}";
        let mut gemma_rejected = gemma4_runtime("question", schema);
        assert!(!gemma_rejected.accept_bytes(gemma_bad));
        let mut gemma_accepted = gemma4_runtime("question", schema);
        assert!(gemma_accepted.accept_bytes(gemma_good));
        assert!(gemma_accepted.is_accepted());
    }

    #[test]
    fn agentic_grammar_contract_cross_family_todowrite_rejects_null_content() {
        let schema = r#"{
            "type": "object",
            "properties": {
                "todos": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "content": {"type": "string"},
                            "status": {"type": "string"},
                            "priority": {"type": "string"}
                        },
                        "required": ["content", "status", "priority"],
                        "additionalProperties": false
                    }
                }
            },
            "required": ["todos"],
            "additionalProperties": false
        }"#;

        let qwen_bad = b"<function=todowrite>\n<parameter=todos>\n[{\"content\":null,\"status\":\"in_progress\",\"priority\":\"high\"}]\n</parameter>\n</function>";
        let qwen_good = b"<function=todowrite>\n<parameter=todos>\n[{\"content\":\"Inspect the environment\",\"status\":\"in_progress\",\"priority\":\"high\"}]\n</parameter>\n</function>";
        let mut qwen_rejected = qwen35_runtime("todowrite", schema);
        assert!(!qwen_rejected.accept_bytes(qwen_bad));
        let mut qwen_accepted = qwen35_runtime("todowrite", schema);
        assert!(qwen_accepted.accept_bytes(qwen_good));
        assert!(qwen_accepted.is_accepted());

        // The native `tojson` filter places one space after commas and
        // colons. Constrained decoding must admit that exact trained surface;
        // forcing compact JSON here changes the model trajectory and can turn
        // a requested content string into punctuation.
        let qwen_native = b"<function=todowrite>\n<parameter=todos>\n[{\"content\": \"Inspect the environment\", \"status\": \"in_progress\", \"priority\": \"high\"}]\n</parameter>\n</function>";
        let mut qwen_native_accepted = qwen35_runtime("todowrite", schema);
        assert!(qwen_native_accepted.accept_bytes(qwen_native));
        assert!(qwen_native_accepted.is_accepted());

        let gemma_bad = b"call:todowrite{todos:[{content:null,status:<|\"|>in_progress<|\"|>,priority:<|\"|>high<|\"|>}]}";
        let gemma_good = b"call:todowrite{todos:[{content:<|\"|>Inspect the environment<|\"|>,status:<|\"|>in_progress<|\"|>,priority:<|\"|>high<|\"|>}]}";
        let mut gemma_rejected = gemma4_runtime("todowrite", schema);
        assert!(!gemma_rejected.accept_bytes(gemma_bad));
        let mut gemma_accepted = gemma4_runtime("todowrite", schema);
        assert!(gemma_accepted.accept_bytes(gemma_good));
        assert!(gemma_accepted.is_accepted());
    }

    #[test]
    fn deepseek4_question_candidate_mask_matches_runtime_on_malformed_json() {
        let schema = serde_json::json!({
            "type": "object",
            "properties": {"questions": {"type": "array"}},
            "required": ["questions"]
        });
        let grammar = DEEPSEEK4
            .tool_call_gbnf(
                "question",
                &schema,
                GrammarShape::OneOrMoreCallsBodyOnly { parallel: false },
            )
            .expect("question grammar");
        let parsed = crate::serve::api::grammar::parser::parse(&grammar).expect("parse grammar");
        let root = parsed.rule_id("root").expect("root");
        let mut runtime = crate::serve::api::grammar::sampler::GrammarRuntime::new(parsed, root)
            .expect("runtime");
        let malformed = "\n<｜DSML｜invoke name=\"question\">\n<｜DSML｜parameter name=\"questions\" string=\"false\">[{\"header\":null,\"options\":{\"label\":\"Movies\",}</｜DSML｜parameter>\n</｜DSML｜invoke>\n</｜DSML｜tool_calls>";
        let mut rejected_at = None;
        for (offset, ch) in malformed.char_indices() {
            let mut encoded = [0u8; 4];
            let bytes = ch.encode_utf8(&mut encoded).as_bytes().to_vec();
            let oracle_accepts = runtime.clone().accept_bytes(&bytes);
            let mut logits = [0.0f32];
            crate::serve::api::grammar::mask::mask_invalid_tokens(
                &runtime,
                &[bytes.clone()],
                &mut logits,
            );
            assert_eq!(
                logits[0].is_finite(),
                oracle_accepts,
                "candidate mask/runtime mismatch at byte {offset}, char {ch:?}"
            );
            if !oracle_accepts {
                rejected_at = Some((offset, ch));
                break;
            }
            assert!(runtime.accept_bytes(&bytes));
        }
        assert!(
            rejected_at.is_some(),
            "the observed malformed DSML sequence must be masked before completion"
        );
    }

    #[test]
    #[ignore = "requires HF2Q_DEEPSEEK4_TOKENIZER pointing to the exact served tokenizer.json"]
    fn deepseek4_question_candidate_mask_rejects_real_tokenizer_chunks() {
        let tokenizer_path = std::env::var("HF2Q_DEEPSEEK4_TOKENIZER")
            .expect("set HF2Q_DEEPSEEK4_TOKENIZER for this exact-artifact test");
        let tokenizer = tokenizers::Tokenizer::from_file(&tokenizer_path)
            .unwrap_or_else(|error| panic!("load {tokenizer_path}: {error}"));
        let schema = serde_json::json!({
            "type": "object",
            "properties": {"questions": {"type": "array"}},
            "required": ["questions"]
        });
        let grammar = DEEPSEEK4
            .tool_call_gbnf(
                "question",
                &schema,
                GrammarShape::OneOrMoreCallsBodyOnly { parallel: false },
            )
            .expect("question grammar");
        let parsed = crate::serve::api::grammar::parser::parse(&grammar).expect("parse grammar");
        let root = parsed.rule_id("root").expect("root");
        let mut runtime = crate::serve::api::grammar::sampler::GrammarRuntime::new(parsed, root)
            .expect("runtime");
        let malformed = "\n<｜DSML｜invoke name=\"question\">\n<｜DSML｜parameter name=\"questions\" string=\"false\">[{\"header\":null,\"options\":{\"label\":\"Movies\",}</｜DSML｜parameter>\n</｜DSML｜invoke>\n</｜DSML｜tool_calls>";
        let encoding = tokenizer.encode(malformed, false).expect("encode mutant");
        let mut rejected_token = None;
        for &token in encoding.get_ids() {
            let bytes = tokenizer
                .decode(&[token], false)
                .expect("decode token")
                .into_bytes();
            assert!(!bytes.is_empty(), "mutant token {token} decoded empty");
            let oracle_accepts = runtime.clone().accept_bytes(&bytes);
            let mut logits = vec![f32::NEG_INFINITY; token as usize + 1];
            logits[token as usize] = 0.0;
            let mut table = vec![Vec::new(); token as usize + 1];
            table[token as usize] = bytes.clone();
            crate::serve::api::grammar::mask::mask_invalid_tokens(&runtime, &table, &mut logits);
            assert_eq!(
                logits[token as usize].is_finite(),
                oracle_accepts,
                "candidate mask/runtime mismatch for tokenizer token {token}"
            );
            if !oracle_accepts {
                rejected_token = Some(token);
                break;
            }
            assert!(runtime.accept_bytes(&bytes));
        }
        assert!(
            rejected_token.is_some(),
            "the malformed DSML mutant must be rejected on the real token boundaries"
        );
    }

    #[test]
    fn deepseek4_todowrite_candidate_mask_keeps_meaningful_string_chunks() {
        let schema = serde_json::json!({
            "type": "object",
            "properties": {
                "todos": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "content": {"type": "string"},
                            "status": {"type": "string"},
                            "priority": {"type": "string"}
                        },
                        "required": ["content", "status", "priority"],
                        "additionalProperties": false
                    }
                }
            },
            "required": ["todos"],
            "additionalProperties": false
        });
        let grammar = DEEPSEEK4
            .tool_call_gbnf(
                "todowrite",
                &schema,
                GrammarShape::OneOrMoreCallsBodyOnly { parallel: false },
            )
            .expect("todowrite grammar");
        let parsed = crate::serve::api::grammar::parser::parse(&grammar).expect("parse grammar");
        let root = parsed.rule_id("root").expect("root");
        let mut runtime = crate::serve::api::grammar::sampler::GrammarRuntime::new(parsed, root)
            .expect("runtime");
        let prefix = "\n<｜DSML｜invoke name=\"todowrite\">\n<｜DSML｜parameter name=\"todos\" string=\"false\">[{\"content\":\"";
        assert!(runtime.accept_bytes(prefix.as_bytes()), "content prefix");

        let table = ["Inspect", " the", " environment", ",", "\"", "null"]
            .map(|value| value.as_bytes().to_vec());
        let mut logits = vec![1.0_f32; table.len()];
        crate::serve::api::grammar::mask::mask_invalid_tokens(&runtime, &table, &mut logits);
        for (index, value) in table.iter().enumerate() {
            let oracle_accepts = runtime.clone().accept_bytes(value);
            assert_eq!(
                logits[index].is_finite(),
                oracle_accepts,
                "candidate mask/runtime mismatch for {:?}",
                String::from_utf8_lossy(value)
            );
        }
        assert!(logits[0].is_finite(), "word token must remain sampleable");
        assert!(
            logits[1].is_finite(),
            "space-prefixed word must remain sampleable"
        );
        assert!(
            logits[2].is_finite(),
            "long word chunk must remain sampleable"
        );
        assert!(
            logits[3].is_finite(),
            "punctuation is also valid string content"
        );
    }

    #[test]
    #[ignore = "requires HF2Q_DEEPSEEK4_TOKENIZER pointing to the exact served tokenizer.json"]
    fn deepseek4_todowrite_candidate_mask_matches_clone_oracle_for_real_vocabulary() {
        let tokenizer_path = std::env::var("HF2Q_DEEPSEEK4_TOKENIZER")
            .expect("set HF2Q_DEEPSEEK4_TOKENIZER for this exact-artifact test");
        let tokenizer = tokenizers::Tokenizer::from_file(&tokenizer_path)
            .unwrap_or_else(|error| panic!("load {tokenizer_path}: {error}"));
        let schema = serde_json::json!({
            "type": "object",
            "properties": {
                "todos": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "content": {"type": "string"},
                            "status": {"type": "string"},
                            "priority": {"type": "string"}
                        },
                        "required": ["content", "status", "priority"],
                        "additionalProperties": false
                    }
                }
            },
            "required": ["todos"],
            "additionalProperties": false
        });
        let grammar = DEEPSEEK4
            .tool_call_gbnf(
                "todowrite",
                &schema,
                GrammarShape::OneOrMoreCallsBodyOnly { parallel: false },
            )
            .expect("todowrite grammar");
        let parsed = crate::serve::api::grammar::parser::parse(&grammar).expect("parse grammar");
        let root = parsed.rule_id("root").expect("root");
        let mut runtime = crate::serve::api::grammar::sampler::GrammarRuntime::new(parsed, root)
            .expect("runtime");
        let prefix = "\n<｜DSML｜invoke name=\"todowrite\">\n<｜DSML｜parameter name=\"todos\" string=\"false\">[{\"content\":\"";
        assert!(runtime.accept_bytes(prefix.as_bytes()), "content prefix");

        let vocab_size = tokenizer.get_vocab_size(true);
        let table = (0..vocab_size as u32)
            .map(|token| {
                tokenizer
                    .decode(&[token], false)
                    .unwrap_or_default()
                    .into_bytes()
            })
            .collect::<Vec<_>>();
        let mut logits = vec![1.0_f32; table.len()];
        crate::serve::api::grammar::mask::mask_invalid_tokens(&runtime, &table, &mut logits);

        let mut mismatches = Vec::new();
        for (token, bytes) in table.iter().enumerate() {
            let oracle_accepts = bytes.is_empty() || runtime.clone().accept_bytes(bytes);
            if logits[token].is_finite() != oracle_accepts {
                mismatches.push((token, String::from_utf8_lossy(bytes).into_owned()));
                if mismatches.len() == 16 {
                    break;
                }
            }
        }
        assert!(
            mismatches.is_empty(),
            "candidate-set mask diverged from clone oracle: {mismatches:?}"
        );
    }

    #[test]
    fn deepseek4_string_parameter_accepts_source_code_with_angle_brackets() {
        let schema = r#"{
            "type": "object",
            "properties": {
                "content": {"type": "string"}
            },
            "required": ["content"]
        }"#;
        let mut runtime = deepseek4_runtime(
            "write",
            schema,
            GrammarShape::OneOrMoreCalls { parallel: false },
        );
        let call = r#"<｜DSML｜tool_calls>
<｜DSML｜invoke name="write">
<｜DSML｜parameter name="content" string="true">impl fmt::Display for Bottles {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} bottles", self.0)
    }
}</｜DSML｜parameter>
</｜DSML｜invoke>
</｜DSML｜tool_calls>"#;
        assert!(runtime.accept_bytes(call.as_bytes()));
        assert!(runtime.is_accepted());

        let body = call
            .strip_prefix("<｜DSML｜tool_calls>")
            .and_then(|value| value.strip_suffix("</｜DSML｜tool_calls>"))
            .expect("outer DSML block");
        let parsed = parse_tool_call_bodies(&DEEPSEEK4, body).expect("parse DSML tool call");
        assert_eq!(parsed.len(), 1);
        let arguments: serde_json::Value =
            serde_json::from_str(&parsed[0].arguments_json).expect("arguments JSON");
        let content = arguments["content"].as_str().expect("content string");
        assert!(content.contains("fmt::Formatter<'_>"));
        assert!(content.contains("-> fmt::Result"));
    }

    #[test]
    fn deepseek4_parallel_grammar_uses_one_outer_block() {
        let schema = r#"{"type":"object","properties":{}}"#;
        let mut runtime = deepseek4_runtime(
            "ping",
            schema,
            GrammarShape::OneOrMoreCalls { parallel: true },
        );
        let calls = "<｜DSML｜tool_calls>\n<｜DSML｜invoke name=\"ping\">\n\n</｜DSML｜invoke>\n<｜DSML｜invoke name=\"ping\">\n\n</｜DSML｜invoke>\n</｜DSML｜tool_calls>";
        assert!(runtime.accept_bytes(calls.as_bytes()));
        assert!(runtime.is_accepted());
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

    #[test]
    fn gemma4_string_parameter_accepts_source_code_with_angle_brackets() {
        let schema = r#"{
            "type": "object",
            "properties": {
                "content": {"type": "string"}
            },
            "required": ["content"]
        }"#;
        let mut runtime = gemma4_runtime("emit_source", schema);
        let body = r#"call:emit_source{content:<|"|>fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result<|"|>}"#;

        assert!(runtime.accept_bytes(body.as_bytes()));
        assert!(runtime.is_accepted());

        let parsed = parse_gemma4_tool_call(body).expect("source tool call must parse");
        let arguments: serde_json::Value =
            serde_json::from_str(&parsed.arguments_json).expect("arguments JSON");
        assert_eq!(
            arguments["content"],
            "fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result"
        );
    }

    #[test]
    fn qwen35_string_parameter_accepts_source_code_with_angle_brackets() {
        let schema = r#"{
            "type": "object",
            "properties": {
                "content": {"type": "string"}
            },
            "required": ["content"]
        }"#;
        let mut runtime = qwen35_runtime("emit_source", schema);
        let body = "<function=emit_source>\n<parameter=content>\nfn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result\n</parameter>\n</function>";

        assert!(runtime.accept_bytes(body.as_bytes()));
        assert!(runtime.is_accepted());

        let parsed = parse_qwen35_tool_call(body).expect("source tool call must parse");
        let arguments: serde_json::Value =
            serde_json::from_str(&parsed.arguments_json).expect("arguments JSON");
        assert_eq!(
            arguments["content"],
            "fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result"
        );
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
        assert!(
            !(ok && rt.is_accepted()),
            "malformed prefix accepted (should reject)"
        );
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
        assert!(
            !(ok && rt.is_accepted()),
            "wrong delimiter accepted (should reject)"
        );
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

    /// iter-219b grammar-exhaust regression guard (2026-05-01) — Agent C's
    /// audit found that `space ::= | " " | "\n"{1,2} [ \t]{0,20}` keeps
    /// `is_dead` from ever flipping after the close marker is consumed
    /// (empty alt + `[ \t]{0,20}` min 0 → runtime perpetually accepting
    /// AND alive). iter-218 claimed "natural grammar-exhaust" termination
    /// for `OneOrMoreCallsBodyOnly{parallel:false}`, but the engine's
    /// break gate at `engine.rs:5045-5059` (`is_accepted && bytes.is_empty()`)
    /// only fires for empty-byte tokens; `<tool_call|>` decodes to 12
    /// bytes (non-empty) so the loop runs past close marker until
    /// max_tokens. This test pins the contract:
    ///
    ///   After accepting `<body><close_marker>`, the grammar MUST be in
    ///   a TERMINAL state — any subsequent byte must terminate (kill all
    ///   stacks; `is_dead == true`). Trailing whitespace allowance was
    ///   the source of the close-gate gap.
    ///
    /// PRE-FIX HEAD: this test FAILS at the `assert!(!alive)` line
    /// because `space`'s `\n{1,2}` alt accepts the trailing newline.
    /// POST-FIX (drop trailing `space` from `OneOrMoreCallsBodyOnly{false}`
    /// + `SingleBody` root_body): this test PASSES.
    #[test]
    fn iter219b_grammar_exhaust_after_close_marker_is_terminal() {
        let schema_json = r#"{"type":"object","properties":{"x":{"type":"integer"}}}"#;
        let schema_v: serde_json::Value = serde_json::from_str(schema_json).unwrap();
        let gbnf = GEMMA4
            .tool_call_gbnf(
                "f",
                &schema_v,
                GrammarShape::OneOrMoreCallsBodyOnly { parallel: false },
            )
            .expect("tool_call_gbnf");
        let mut rt = grammar_runtime_for_gbnf(&gbnf);
        // Accept canonical body + close marker per the lazy-gate contract
        // (open marker has been stripped from the grammar — consumed by the
        // awaiting_trigger no-op gate before grammar engagement).
        assert!(
            rt.accept_bytes(b"call:f{x:1}<tool_call|>"),
            "canonical body+close rejected"
        );
        assert!(
            rt.is_accepted(),
            "post-close: rule must be in an accepting state"
        );
        // Contract: any further byte must terminate the grammar. iter-218's
        // claimed grammar-exhaust termination relies on this — without it,
        // the decode loop runs past close marker because the engine's
        // `is_accepted && bytes.is_empty()` break gate only fires for
        // empty-byte tokens and `<tool_call|>` decodes to 12 bytes.
        let mut clone_lf = rt.clone();
        let alive_after_lf = clone_lf.accept_bytes(b"\n");
        assert!(
            !alive_after_lf,
            "iter-219b: grammar must terminate on trailing `\\n` after close \
             marker; HEAD's `space ::= | \" \" | \"\\n\"{{1,2}} [ \\t]{{0,20}}` \
             allows up to 22 trailing whitespace bytes which prevents \
             `is_dead` from flipping. Drop the trailing ` space` from \
             OneOrMoreCallsBodyOnly{{false}} + SingleBody root_body."
        );
        assert!(
            clone_lf.is_dead(),
            "is_dead must flip to true after rejected continuation post-close"
        );
        // Same contract for a non-whitespace byte (start of stray
        // `<|tool_response>` from the LIVE bug surface).
        let mut clone_lt = rt.clone();
        let alive_after_lt = clone_lt.accept_bytes(b"<");
        assert!(
            !alive_after_lt,
            "iter-219b: post-close grammar must reject `<` (start of any \
             special-token leak like <|tool_response>)"
        );
        assert!(
            clone_lt.is_dead(),
            "is_dead must flip on rejected `<` after close"
        );
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
        let input =
            b"[function=get_weather]\n[parameter=location]\nParis\n[/parameter]\n[/function]";
        let ok = rt.accept_bytes(input);
        assert!(
            !(ok && rt.is_accepted()),
            "malformed wrapper accepted (should reject)"
        );
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
        let result = unknown.tool_call_gbnf("f", &schema, GrammarShape::SingleBody);
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
        assert!(
            rt.accept_bytes(input),
            "required key present should be accepted"
        );
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
        assert!(
            !(ok && rt.is_accepted()),
            "missing required key must be rejected"
        );
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
        assert!(
            rt.accept_bytes(input),
            "permuted required keys must be accepted"
        );
        assert!(rt.is_accepted());
    }

    /// B1 Gemma4: schema with 9 required keys → EmitterError::TooManyRequiredKeys.
    /// W-ζ: cap lowered from 16 to 8 to match json_schema.rs ANY_ORDER_MAX_REQUIRED.
    #[test]
    fn b1_gemma4_too_many_required_keys_err() {
        // Build a schema with 13 required keys (> MAX_REQUIRED_KEYS = 12).
        let mut props = serde_json::Map::new();
        let mut required = Vec::new();
        for i in 0..13usize {
            let k = format!("key{}", i);
            props.insert(k.clone(), serde_json::json!({"type": "string"}));
            required.push(serde_json::Value::String(k));
        }
        let schema = serde_json::json!({
            "type": "object",
            "properties": props,
            "required": required
        });
        let result = GEMMA4.tool_call_gbnf("f", &schema, GrammarShape::SingleBody);
        assert!(result.is_err(), "13 required keys must return Err");
        let msg = result.unwrap_err();
        assert!(
            msg.contains("13"),
            "error message should mention count: {}",
            msg
        );
        assert!(
            msg.contains("12"),
            "error message should mention cap: {}",
            msg
        );
    }

    /// ADR-052 boundary — exactly 12 required keys compiles OK.
    #[test]
    fn thirteen_required_keys_in_gemma_tool_call_gbnf_returns_too_many_required_keys() {
        // 13 keys — must be rejected (boundary at 12).
        let mut props = serde_json::Map::new();
        let mut required = Vec::new();
        for i in 0..13usize {
            let k = format!("p{}", i);
            props.insert(k.clone(), serde_json::json!({"type": "integer"}));
            required.push(serde_json::Value::String(k));
        }
        let schema = serde_json::json!({
            "type": "object",
            "properties": props,
            "required": required
        });
        let result = GEMMA4.tool_call_gbnf("tool13", &schema, GrammarShape::SingleBody);
        assert!(
            result.is_err(),
            "13 required keys must return TooManyRequiredKeys"
        );
        let msg = result.unwrap_err();
        assert!(msg.contains("13"), "error must mention count 13: {}", msg);
        assert!(msg.contains("12"), "error must mention cap 12: {}", msg);
    }

    /// ADR-052 boundary — exactly 12 required keys compiles OK.
    #[test]
    fn twelve_required_keys_in_gemma_tool_call_gbnf_compiles_ok() {
        let mut props = serde_json::Map::new();
        let mut required = Vec::new();
        for i in 0..12usize {
            let k = format!("p{}", i);
            props.insert(k.clone(), serde_json::json!({"type": "integer"}));
            required.push(serde_json::Value::String(k));
        }
        let schema = serde_json::json!({
            "type": "object",
            "properties": props,
            "required": required
        });
        let result = GEMMA4.tool_call_gbnf("tool12", &schema, GrammarShape::SingleBody);
        assert!(
            result.is_ok(),
            "12 required keys must compile without error"
        );
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
        assert!(
            rt.accept_bytes(input),
            "required key present should be accepted"
        );
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
        assert!(
            !(ok && rt.is_accepted()),
            "missing required key must be rejected"
        );
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
        assert!(
            rt.accept_bytes(input),
            "permuted required keys must be accepted"
        );
        assert!(rt.is_accepted());
    }

    /// B1 Qwen35: 13 required keys → EmitterError::TooManyRequiredKeys.
    #[test]
    fn b1_qwen35_too_many_required_keys_err() {
        let mut props = serde_json::Map::new();
        let mut required = Vec::new();
        for i in 0..13usize {
            let k = format!("key{}", i);
            props.insert(k.clone(), serde_json::json!({"type": "string"}));
            required.push(serde_json::Value::String(k));
        }
        let schema = serde_json::json!({
            "type": "object",
            "properties": props,
            "required": required
        });
        let result = QWEN35.tool_call_gbnf("f", &schema, GrammarShape::SingleBody);
        assert!(result.is_err(), "13 required keys must return Err");
        let msg = result.unwrap_err();
        assert!(
            msg.contains("13"),
            "error message should mention count: {}",
            msg
        );
        assert!(
            msg.contains("12"),
            "error message should mention cap: {}",
            msg
        );
    }

    /// ADR-052 boundary — 13 required keys returns TooManyRequiredKeys.
    #[test]
    fn thirteen_required_keys_in_qwen35_tool_call_gbnf_returns_too_many_required_keys() {
        let mut props = serde_json::Map::new();
        let mut required = Vec::new();
        for i in 0..13usize {
            let k = format!("q{}", i);
            props.insert(k.clone(), serde_json::json!({"type": "string"}));
            required.push(serde_json::Value::String(k));
        }
        let schema = serde_json::json!({
            "type": "object",
            "properties": props,
            "required": required
        });
        let result = QWEN35.tool_call_gbnf("qtool13", &schema, GrammarShape::SingleBody);
        assert!(
            result.is_err(),
            "13 required keys must return TooManyRequiredKeys"
        );
        let msg = result.unwrap_err();
        assert!(msg.contains("13"), "error must mention count 13: {}", msg);
        assert!(msg.contains("12"), "error must mention cap 12: {}", msg);
    }

    /// ADR-052 boundary — exactly 12 required keys compiles OK.
    #[test]
    fn twelve_required_keys_in_qwen35_tool_call_gbnf_compiles_ok() {
        let mut props = serde_json::Map::new();
        let mut required = Vec::new();
        for i in 0..12usize {
            let k = format!("q{}", i);
            props.insert(k.clone(), serde_json::json!({"type": "string"}));
            required.push(serde_json::Value::String(k));
        }
        let schema = serde_json::json!({
            "type": "object",
            "properties": props,
            "required": required
        });
        let result = QWEN35.tool_call_gbnf("qtool12", &schema, GrammarShape::SingleBody);
        assert!(
            result.is_ok(),
            "12 required keys must compile without error"
        );
    }

    // -----------------------------------------------------------------------
    // iter-231a (supersedes Wave 2.5 B3) — structured-parameter acceptance
    //
    // Pre-iter-231a these pinned a hard `EmitterError::UnsupportedSchema`
    // rejection for `array`/`object` params (→ HTTP 400 for the whole
    // request).  That gate broke real-world tool schemas (MCP servers
    // expose free-form `object` params such as `config`, `arguments`).
    // The grammar now compiles such params to permissive recursive value
    // rules and ACCEPTS any well-formed structured value.
    // -----------------------------------------------------------------------

    /// iter-231a Gemma4: `array` param compiles AND the runtime accepts a
    /// nested kv-list value exactly as `format_argument` renders it.
    #[test]
    fn iter231a_gemma4_array_param_compiles_and_accepts_nested_value() {
        let schema = r#"{
            "type": "object",
            "properties": {
                "tags": {"type": "array"},
                "name": {"type": "string"}
            }
        }"#;
        let mut rt = gemma4_runtime("tag_item", schema);
        // format_argument: strings `<|"|>...<|"|>`, scalars bare, nested
        // containers recursive, keys bare (escape_keys=False).
        let input = b"call:tag_item{name:<|\"|>x<|\"|>,tags:[<|\"|>a<|\"|>,<|\"|>b<|\"|>,1,true,null,[2,{k:<|\"|>v<|\"|>}]]}";
        assert!(rt.accept_bytes(input), "nested array value rejected");
        assert!(rt.is_accepted(), "not accepted at end");
    }

    /// iter-231a Gemma4: `object` param compiles AND accepts a nested
    /// object with bare keys at every level (escape_keys=False).
    #[test]
    fn iter231a_gemma4_object_param_compiles_and_accepts_nested_value() {
        let schema = r#"{
            "type": "object",
            "properties": {
                "config": {"type": "object"}
            }
        }"#;
        let mut rt = gemma4_runtime("configure", schema);
        let input = b"call:configure{config:{model:<|\"|>sonnet<|\"|>,retries:3,nested:{on:true,ids:[1,2]}}}";
        assert!(rt.accept_bytes(input), "nested object value rejected");
        assert!(rt.is_accepted(), "not accepted at end");
    }

    /// iter-231a Qwen35: `array` param compiles AND accepts nested compact
    /// JSON (what the template's `tojson` filter emits).
    #[test]
    fn iter231a_qwen35_array_param_compiles_and_accepts_nested_value() {
        let schema = r#"{
            "type": "object",
            "properties": {
                "tags": {"type": "array"}
            }
        }"#;
        let mut rt = qwen35_runtime("tag_item", schema);
        let input = b"<function=tag_item>\n<parameter=tags>\n[\"a\",\"b\",1,true,null,[2,{\"k\":\"v\"}]]\n</parameter>\n</function>";
        assert!(rt.accept_bytes(input), "nested JSON array rejected");
        assert!(rt.is_accepted(), "not accepted at end");
    }

    /// iter-231a Qwen35: `object` param compiles AND accepts nested
    /// compact JSON with mixed scalar/container children.
    #[test]
    fn iter231a_qwen35_object_param_compiles_and_accepts_nested_value() {
        let schema = r#"{
            "type": "object",
            "properties": {
                "metadata": {"type": "object"}
            }
        }"#;
        let mut rt = qwen35_runtime("set_meta", schema);
        let input = b"<function=set_meta>\n<parameter=metadata>\n{\"model\":\"sonnet\",\"retries\":3,\"nested\":{\"on\":true,\"ids\":[1,2]}}\n</parameter>\n</function>";
        assert!(rt.accept_bytes(input), "nested JSON object rejected");
        assert!(rt.is_accepted(), "not accepted at end");
    }

    /// iter-231a regression — the exact failure shape that motivated the
    /// supersession: an MCP-style tool schema with a free-form `object`
    /// parameter (`config`) plus scalars and an enum.  Pre-iter-231a this
    /// 400'd the entire request at grammar-compile time; opencode and
    /// other agentic clients could not call ANY tool once such a schema
    /// was present in tools[].
    #[test]
    fn iter231a_qwen35_mcp_tool_with_freeform_object_param_compiles_and_runs() {
        let schema = r#"{
            "type": "object",
            "properties": {
                "agentType": {"type": "string"},
                "config": {"type": "object"},
                "memoryDimension": {"type": "integer"},
                "model": {"type": "string", "enum": ["haiku", "sonnet", "opus"]},
                "task": {"type": "string"}
            }
        }"#;
        let mut rt = qwen35_runtime("agent_spawn", schema);
        let input = b"<function=agent_spawn>\n<parameter=agentType>\ncoder\n</parameter>\n<parameter=config>\n{\"maxTurns\":50,\"env\":{\"KEY\":\"value\"},\"nested\":[1,{\"x\":true}]}\n</parameter>\n<parameter=memoryDimension>\n384\n</parameter>\n<parameter=model>\nsonnet\n</parameter>\n<parameter=task>\nimplement feature\n</parameter>\n</function>";
        assert!(rt.accept_bytes(input), "MCP-shaped call rejected");
        assert!(rt.is_accepted(), "not accepted at end");
    }

    /// JSON string values must preserve source-shaped angle brackets. `<` is
    /// a valid unescaped JSON character; the enclosing JSON grammar prevents
    /// it from consuming the parameter close tag.
    #[test]
    fn iter231a_qwen35_object_param_accepts_angle_bracket_in_json_string() {
        let schema = r#"{
            "type": "object",
            "properties": {
                "config": {"type": "object"}
            }
        }"#;
        let mut rt = qwen35_runtime("configure", schema);
        let input = b"<function=configure>\n<parameter=config>\n{\"a\":\"<bad\"}\n</parameter>\n</function>";
        assert!(rt.accept_bytes(input));
        assert!(rt.is_accepted());
    }

    /// iter-231a integration: scalar params still compile and run under
    /// both emitters (supersession must not regress scalars).
    #[test]
    fn iter231a_scalar_params_unaffected() {
        let schema = serde_json::json!({
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "count": {"type": "integer"},
                "enabled": {"type": "boolean"}
            }
        });
        assert!(
            GEMMA4
                .tool_call_gbnf("f", &schema, GrammarShape::SingleBody)
                .is_ok(),
            "scalars must compile"
        );
        assert!(
            QWEN35
                .tool_call_gbnf("f", &schema, GrammarShape::SingleBody)
                .is_ok(),
            "scalars must compile"
        );
    }

    /// iter-231d: OpenCode plugin tools built from Zod records emit the
    /// semantically-redundant `propertyNames:{type:"string"}` keyword. JSON
    /// object keys are already strings, so both native tool surfaces must
    /// compile the schema and preserve arbitrary nested argument values.
    #[test]
    fn agentic_grammar_contract_tautological_property_names_compiles_and_runs() {
        let schema = r#"{
            "type": "object",
            "properties": {
                "arguments": {
                    "type": "object",
                    "propertyNames": {"type": "string"},
                    "additionalProperties": {}
                }
            },
            "required": ["arguments"]
        }"#;

        let mut qwen = qwen35_runtime("ruflo_call", schema);
        let qwen_call = b"<function=ruflo_call>\n<parameter=arguments>\n{\"query\":\"cache\",\"limit\":5,\"nested\":{\"ok\":true}}\n</parameter>\n</function>";
        assert!(
            qwen.accept_bytes(qwen_call),
            "Qwen rejected OpenCode Zod record"
        );
        assert!(qwen.is_accepted(), "Qwen record call incomplete");

        let mut gemma = gemma4_runtime("ruflo_call", schema);
        let gemma_call =
            b"call:ruflo_call{arguments:{query:<|\"|>cache<|\"|>,limit:5,nested:{ok:true}}}";
        assert!(
            gemma.accept_bytes(gemma_call),
            "Gemma rejected OpenCode Zod record"
        );
        assert!(gemma.is_accepted(), "Gemma record call incomplete");
    }

    /// `propertyNames` is ignored only when it adds no constraint beyond the
    /// JSON object-key type. Key patterns/enums/length limits remain an honest
    /// compilation error instead of silently widening the accepted language.
    #[test]
    fn agentic_grammar_contract_constrained_property_names_fails_closed() {
        let schema = serde_json::json!({
            "type": "object",
            "properties": {
                "arguments": {
                    "type": "object",
                    "propertyNames": {"type": "string", "pattern": "^[a-z]+$"}
                }
            }
        });
        for registration in [&QWEN35, &GEMMA4] {
            let error = registration
                .tool_call_gbnf("ruflo_call", &schema, GrammarShape::SingleBody)
                .expect_err("constrained propertyNames must fail closed");
            assert!(
                error.contains("constrained propertyNames"),
                "error must name the unsupported constraint: {error}"
            );
            assert!(
                error.contains("/arguments"),
                "error must carry the path: {error}"
            );
        }
    }

    #[test]
    fn agentic_grammar_contract_root_property_names_uses_fail_closed_boundary() {
        for property_names in [
            serde_json::json!(true),
            serde_json::json!({}),
            serde_json::json!({"type": "string"}),
        ] {
            let schema = serde_json::json!({
                "type": "object",
                "properties": {},
                "propertyNames": property_names,
            });
            for registration in [&QWEN35, &GEMMA4] {
                registration
                    .tool_call_gbnf("ruflo_call", &schema, GrammarShape::SingleBody)
                    .expect("unconstrained root propertyNames must compile");
            }
        }

        let constrained = serde_json::json!({
            "type": "object",
            "properties": {},
            "propertyNames": {"type": "string", "minLength": 1},
        });
        for registration in [&QWEN35, &GEMMA4] {
            let error = registration
                .tool_call_gbnf("ruflo_call", &constrained, GrammarShape::SingleBody)
                .expect_err("constrained root propertyNames must fail closed");
            assert!(error.contains("constrained propertyNames"));
        }
    }

    // -----------------------------------------------------------------------
    // iter-231b — full-fidelity nested-schema compilation
    //
    // Structured parameters now constrain everything the schema DECLARES
    // (properties, required, typed items, enums, unions, additionalProperties)
    // and stay open only where the schema itself is open (free-form
    // object/array → permissive recursive value).  Same contract as the
    // top-level parameter grammar: required keys in any order first, then
    // optional keys in any order.
    // -----------------------------------------------------------------------

    /// Qwen35: nested object with declared properties — required keys are
    /// accepted in ANY order (permutation grammar), compact tojson form.
    #[test]
    fn iter231b_qwen35_nested_object_required_keys_any_order() {
        let schema = r#"{
            "type": "object",
            "properties": {
                "server": {
                    "type": "object",
                    "properties": {
                        "host": {"type": "string"},
                        "port": {"type": "integer"},
                        "tls": {"type": "boolean"}
                    },
                    "required": ["host", "port"]
                }
            }
        }"#;
        // host before port.
        let mut rt = qwen35_runtime("connect", schema);
        let input = b"<function=connect>\n<parameter=server>\n{\"host\":\"example.com\",\"port\":443}\n</parameter>\n</function>";
        assert!(rt.accept_bytes(input), "declared-order rejected");
        assert!(rt.is_accepted());
        // port before host (required any-order), plus optional tls.
        let mut rt2 = qwen35_runtime("connect", schema);
        let input2 = b"<function=connect>\n<parameter=server>\n{\"port\":443,\"host\":\"example.com\",\"tls\":true}\n</parameter>\n</function>";
        assert!(rt2.accept_bytes(input2), "reversed required order rejected");
        assert!(rt2.is_accepted());
    }

    /// Qwen35: omitting a nested required key must be REJECTED by the
    /// grammar (the model physically cannot emit that token path).
    #[test]
    fn iter231b_qwen35_nested_object_missing_required_rejected() {
        let schema = r#"{
            "type": "object",
            "properties": {
                "server": {
                    "type": "object",
                    "properties": {
                        "host": {"type": "string"},
                        "port": {"type": "integer"}
                    },
                    "required": ["host", "port"]
                }
            }
        }"#;
        let mut rt = qwen35_runtime("connect", schema);
        // port missing — the permutation grammar can never complete.
        let input = b"<function=connect>\n<parameter=server>\n{\"host\":\"example.com\"}\n</parameter>\n</function>";
        let ok = rt.accept_bytes(input);
        assert!(
            !(ok && rt.is_accepted()),
            "nested object missing a required key accepted (must reject)"
        );
    }

    /// Qwen35: `additionalProperties:false` closes the nested key set —
    /// an undeclared key is rejected; the same shape WITHOUT the flag
    /// (open, JSON Schema default) accepts it via the wildcard kv tail.
    #[test]
    fn iter231b_qwen35_nested_object_additional_properties_gate() {
        let closed = r#"{
            "type": "object",
            "properties": {
                "cfg": {
                    "type": "object",
                    "properties": {"a": {"type": "integer"}},
                    "required": ["a"],
                    "additionalProperties": false
                }
            }
        }"#;
        let mut rt = qwen35_runtime("f", closed);
        let input =
            b"<function=f>\n<parameter=cfg>\n{\"a\":1,\"zzz\":2}\n</parameter>\n</function>";
        let ok = rt.accept_bytes(input);
        assert!(
            !(ok && rt.is_accepted()),
            "additionalProperties:false accepted an undeclared key"
        );

        let open = r#"{
            "type": "object",
            "properties": {
                "cfg": {
                    "type": "object",
                    "properties": {"a": {"type": "integer"}},
                    "required": ["a"]
                }
            }
        }"#;
        let mut rt2 = qwen35_runtime("f", open);
        assert!(
            rt2.accept_bytes(input),
            "open object rejected an extra key (wildcard tail must accept)"
        );
        assert!(rt2.is_accepted());
    }

    /// Qwen35: typed array items are enforced — `items:{type:"string"}`
    /// accepts `["a","b"]` and rejects `[1,2]`.
    #[test]
    fn iter231b_qwen35_nested_array_typed_items_enforced() {
        let schema = r#"{
            "type": "object",
            "properties": {
                "tags": {"type": "array", "items": {"type": "string"}}
            }
        }"#;
        let mut rt = qwen35_runtime("tag", schema);
        let good = b"<function=tag>\n<parameter=tags>\n[\"a\",\"b\"]\n</parameter>\n</function>";
        assert!(rt.accept_bytes(good), "string items rejected");
        assert!(rt.is_accepted());

        let mut rt2 = qwen35_runtime("tag", schema);
        let bad = b"<function=tag>\n<parameter=tags>\n[1,2]\n</parameter>\n</function>";
        let ok = rt2.accept_bytes(bad);
        assert!(
            !(ok && rt2.is_accepted()),
            "array with items:string accepted integer items (must reject)"
        );
    }

    /// Qwen35: nested enum + anyOf union bodies are enforced.
    ///
    /// The closed object also proves that no undeclared key can enter.
    #[test]
    fn iter231b_qwen35_nested_enum_and_anyof() {
        let schema = r#"{
            "type": "object",
            "properties": {
                "cfg": {
                    "type": "object",
                    "properties": {
                        "mode": {"enum": ["fast", "safe"]},
                        "limit": {"anyOf": [{"type": "integer"}, {"type": "null"}]}
                    },
                    "required": ["mode"],
                    "additionalProperties": false
                }
            }
        }"#;
        let mut rt = qwen35_runtime("f", schema);
        let good = b"<function=f>\n<parameter=cfg>\n{\"mode\":\"fast\",\"limit\":3}\n</parameter>\n</function>";
        assert!(rt.accept_bytes(good), "enum+anyOf valid value rejected");
        assert!(rt.is_accepted());

        let mut rt2 = qwen35_runtime("f", schema);
        // "warp" is not in the enum.
        let bad = b"<function=f>\n<parameter=cfg>\n{\"mode\":\"warp\"}\n</parameter>\n</function>";
        let ok = rt2.accept_bytes(bad);
        assert!(!(ok && rt2.is_accepted()), "undeclared enum value accepted");

        let mut rt3 = qwen35_runtime("f", schema);
        // "3.5" matches neither integer nor null branch of the anyOf, and
        // the object is CLOSED — no wildcard tail to rescue it.
        let bad2 = b"<function=f>\n<parameter=cfg>\n{\"mode\":\"safe\",\"limit\":3.5}\n</parameter>\n</function>";
        let ok2 = rt3.accept_bytes(bad2);
        assert!(
            !(ok2 && rt3.is_accepted()),
            "anyOf-mismatched value accepted on closed object"
        );
    }

    /// Qwen35: an OPEN object's wildcard accepts undeclared keys but cannot
    /// re-match a declared optional key, including via JSON unicode escapes.
    #[test]
    fn iter231b_qwen35_open_object_wildcard_excludes_declared_keys() {
        let schema = r#"{
            "type": "object",
            "properties": {
                "cfg": {
                    "type": "object",
                    "properties": {
                        "mode": {"enum": ["fast", "safe"]},
                        "limit": {"type": "integer"}
                    },
                    "required": ["mode"]
                }
            }
        }"#;
        let mut rt = qwen35_runtime("f", schema);
        let input = b"<function=f>\n<parameter=cfg>\n{\"mode\":\"safe\",\"limit\":3.5}\n</parameter>\n</function>";
        assert!(
            !rt.accept_bytes(input) || !rt.is_accepted(),
            "open-object wildcard re-matched a declared key"
        );
        let mut escaped = qwen35_runtime("f", schema);
        let escaped_input = b"<function=f>\n<parameter=cfg>\n{\"mode\":\"safe\",\"\\u006cimit\":3.5}\n</parameter>\n</function>";
        assert!(
            !escaped.accept_bytes(escaped_input) || !escaped.is_accepted(),
            "unicode-escaped declared key bypassed the exclusion trie"
        );
        let mut extra = qwen35_runtime("f", schema);
        let extra_input = b"<function=f>\n<parameter=cfg>\n{\"mode\":\"safe\",\"note\":3.5}\n</parameter>\n</function>";
        assert!(extra.accept_bytes(extra_input) && extra.is_accepted());
        // But a REQUIRED key mismatch is still physically impossible:
        let mut rt2 = qwen35_runtime("f", schema);
        let bad = b"<function=f>\n<parameter=cfg>\n{\"mode\":\"warp\",\"limit\":3}\n</parameter>\n</function>";
        assert!(
            !(rt2.accept_bytes(bad) && rt2.is_accepted()),
            "required-key enum mismatch must reject even on open objects"
        );
    }

    /// Qwen35: features the grammar cannot enforce are an honest 400
    /// (`UnsupportedSchemaFeature` naming feature + dot-path), not a
    /// silent permissive downgrade.
    #[test]
    fn iter231b_qwen35_unsupported_nested_feature_errors() {
        let schema = serde_json::json!({
            "type": "object",
            "properties": {
                "cfg": {"type": "object", "properties": {"x": {"$ref": "#/$defs/t"}}}
            }
        });
        let err = QWEN35
            .tool_call_gbnf("f", &schema, GrammarShape::SingleBody)
            .expect_err("$ref must error");
        let msg = err.to_string();
        assert!(msg.contains("$ref"), "error must name the feature: {}", msg);
        assert!(
            msg.contains("/cfg/properties/x"),
            "error must carry the dot-path: {}",
            msg
        );
    }

    /// Qwen35: >8 nested required keys → TooManyRequiredKeys (same SOTA
    /// O(2^N) bound as the top level).
    #[test]
    fn iter231b_qwen35_nested_thirteen_required_keys_errors() {
        let props: serde_json::Map<String, serde_json::Value> = (0..13)
            .map(|i| (format!("k{}", i), serde_json::json!({"type": "integer"})))
            .collect();
        let schema = serde_json::json!({
            "type": "object",
            "properties": {
                "cfg": {
                    "type": "object",
                    "properties": props,
                    "required": ["k0","k1","k2","k3","k4","k5","k6","k7","k8","k9","k10","k11","k12"]
                }
            }
        });
        // Public API converts EmitterError → String; assert the payload.
        let err = QWEN35
            .tool_call_gbnf("f", &schema, GrammarShape::SingleBody)
            .expect_err("13 nested required keys must error");
        assert!(
            err.contains("13 required parameters"),
            "error must carry the required-key count: {}",
            err
        );
        assert!(
            err.contains("/cfg"),
            "error must carry the nested path: {}",
            err
        );
    }

    /// Gemma4: nested object with declared properties — required keys in
    /// any order (bare-key kv surface).
    #[test]
    fn iter231b_gemma4_nested_object_required_keys_any_order() {
        let schema = r#"{
            "type": "object",
            "properties": {
                "server": {
                    "type": "object",
                    "properties": {
                        "host": {"type": "string"},
                        "port": {"type": "integer"}
                    },
                    "required": ["host", "port"]
                }
            }
        }"#;
        let mut rt = gemma4_runtime("connect", schema);
        let input = b"call:connect{server:{port:443,host:<|\"|>example.com<|\"|>}}";
        assert!(rt.accept_bytes(input), "reversed required order rejected");
        assert!(rt.is_accepted());
    }

    /// Gemma4: nested missing required key rejected; typed array items
    /// enforced on the kv surface.
    #[test]
    fn iter231b_gemma4_nested_missing_required_and_typed_items_rejected() {
        let schema = r#"{
            "type": "object",
            "properties": {
                "server": {
                    "type": "object",
                    "properties": {
                        "host": {"type": "string"},
                        "port": {"type": "integer"}
                    },
                    "required": ["host", "port"]
                },
                "tags": {"type": "array", "items": {"type": "string"}}
            }
        }"#;
        let mut rt = gemma4_runtime("connect", schema);
        // port missing.
        let input = b"call:connect{server:{host:<|\"|>example.com<|\"|>}}";
        let ok = rt.accept_bytes(input);
        assert!(
            !(ok && rt.is_accepted()),
            "missing nested required accepted"
        );

        let mut rt2 = gemma4_runtime("connect", schema);
        // integer items against items:string.
        let input2 = b"call:connect{server:{host:<|\"|>h<|\"|>,port:1},tags:[1,2]}";
        let ok2 = rt2.accept_bytes(input2);
        assert!(
            !(ok2 && rt2.is_accepted()),
            "typed array accepted wrong item type"
        );
    }

    /// Gemma4 parser (iter-231b): nested structured arguments round-trip
    /// into `arguments_json` as real JSON objects/arrays — the
    /// pre-iter-231b string-only kv splitter mangled any nested value
    /// containing a comma.
    #[test]
    fn iter231b_gemma4_parser_nested_arguments_round_trip() {
        let body = "call:configure{config:{model:<|\"|>sonnet<|\"|>,retries:3,nested:{on:true,ids:[1,2]}},task:<|\"|>do it<|\"|>}";
        let parsed = parse_gemma4_tool_call(body).expect("nested call must parse");
        assert_eq!(parsed.name, "configure");
        let args: serde_json::Value =
            serde_json::from_str(&parsed.arguments_json).expect("arguments_json valid JSON");
        assert_eq!(
            args,
            serde_json::json!({
                "config": {
                    "model": "sonnet",
                    "retries": 3,
                    "nested": {"on": true, "ids": [1, 2]}
                },
                "task": "do it"
            })
        );
    }

    /// Gemma4 parser (iter-231b): a comma INSIDE a nested container or a
    /// quoted string must not split the top-level kv-list.
    #[test]
    fn iter231b_gemma4_parser_commas_inside_nested_values_do_not_split() {
        let body = "call:f{tags:[<|\"|>a,b<|\"|>,2],note:<|\"|>x,y<|\"|>}";
        let parsed = parse_gemma4_tool_call(body).expect("must parse");
        let args: serde_json::Value =
            serde_json::from_str(&parsed.arguments_json).expect("arguments_json valid JSON");
        assert_eq!(
            args,
            serde_json::json!({
                "tags": ["a,b", 2],
                "note": "x,y"
            })
        );
    }

    /// Qwen35 parser sanity: nested JSON values already round-trip via
    /// the JSON-first value decode (no iter-231b change needed there —
    /// pin the contract so the two families stay symmetric).
    #[test]
    fn iter231b_qwen35_parser_nested_arguments_round_trip() {
        let body = "<function=configure>\n<parameter=config>\n{\"model\":\"sonnet\",\"nested\":{\"ids\":[1,2]}}\n</parameter>\n</function>";
        let parsed = parse_qwen35_tool_call(body).expect("nested call must parse");
        assert_eq!(parsed.name, "configure");
        let args: serde_json::Value =
            serde_json::from_str(&parsed.arguments_json).expect("arguments_json valid JSON");
        assert_eq!(
            args,
            serde_json::json!({"config": {"model": "sonnet", "nested": {"ids": [1, 2]}}})
        );
    }

    #[test]
    fn agentic_grammar_contract_qwen_wire_kinds_preserve_json_looking_strings() {
        let tools = vec![crate::serve::api::schema::Tool {
            tool_type: "function".to_string(),
            function: crate::serve::api::schema::ToolFunction {
                name: "gateway".to_string(),
                description: None,
                parameters: Some(serde_json::json!({
                    "type": "object",
                    "properties": {
                        "arguments_json": {"type": "string"},
                        "enabled": {"type": "boolean"},
                        "count": {"type": "integer"},
                        "payload": {"type": "object"}
                    }
                })),
            },
        }];
        let kinds = qwen35_tool_argument_wire_kinds(&tools).expect("wire kinds");
        assert_eq!(
            kinds.kind_for("gateway", "arguments_json"),
            Some(ToolArgumentWireKind::RawString)
        );
        assert_eq!(
            kinds.kind_for("gateway", "enabled"),
            Some(ToolArgumentWireKind::Json)
        );

        let body = concat!(
            "<function=gateway>",
            "<parameter=arguments_json>{\"query\":\"cache\"}</parameter>",
            "<parameter=enabled>true</parameter>",
            "<parameter=count>42</parameter>",
            "<parameter=payload>{\"nested\":[1,2]}</parameter>",
            "</function>"
        );
        let parsed =
            parse_qwen35_tool_call_with_wire_kinds(body, Some(&kinds)).expect("schema-aware parse");
        let args: serde_json::Value = serde_json::from_str(&parsed.arguments_json).unwrap();
        assert_eq!(args["arguments_json"], "{\"query\":\"cache\"}");
        assert_eq!(args["enabled"], true);
        assert_eq!(args["count"], 42);
        assert_eq!(args["payload"], serde_json::json!({"nested": [1, 2]}));

        assert!(parse_qwen35_tool_call_with_wire_kinds(
            "<function=gateway><parameter=unknown>x</parameter></function>",
            Some(&kinds)
        )
        .is_none());
        assert!(parse_qwen35_tool_call_with_wire_kinds(
            "<function=gateway><parameter=count>not-json</parameter></function>",
            Some(&kinds)
        )
        .is_none());
    }

    #[test]
    fn agentic_grammar_contract_qwen_wire_classifier_handles_unions_and_duplicates() {
        assert_eq!(
            qwen35_top_level_wire_kind(&serde_json::json!({
                "anyOf": [{"type": "string"}, {"enum": ["a", "b"]}]
            })),
            ToolArgumentWireKind::RawString
        );
        assert_eq!(
            qwen35_top_level_wire_kind(&serde_json::json!({
                "oneOf": [{"type": "string"}, {"type": "integer"}]
            })),
            ToolArgumentWireKind::Infer
        );

        let make_tool = |schema_type: &str| crate::serve::api::schema::Tool {
            tool_type: "function".to_string(),
            function: crate::serve::api::schema::ToolFunction {
                name: "duplicate".to_string(),
                description: None,
                parameters: Some(serde_json::json!({
                    "type": "object",
                    "properties": {"value": {"type": schema_type}}
                })),
            },
        };
        let error = qwen35_tool_argument_wire_kinds(&[make_tool("string"), make_tool("integer")])
            .expect_err("conflicting duplicate schema must fail");
        assert!(error.contains("duplicate tool 'duplicate'"));

        let error = qwen35_tool_argument_wire_kinds(&[make_tool("object"), make_tool("array")])
            .expect_err("same-wire duplicate schema must also fail");
        assert!(error.contains("duplicate tool 'duplicate'"));

        let error = qwen35_tool_argument_wire_kinds(&[make_tool("string"), make_tool("string")])
            .expect_err("identical duplicate declarations remain ambiguous");
        assert!(error.contains("duplicate tool 'duplicate'"));
    }

    /// Both families: top-level UNTYPED parameters now accept structured
    /// values too (the template renders untyped args via tojson /
    /// format_argument as well).
    #[test]
    fn iter231b_untyped_params_accept_structured_values() {
        let schema = r#"{"type": "object", "properties": {"payload": {}}}"#;
        let mut rt = qwen35_runtime("f", schema);
        let q_input =
            b"<function=f>\n<parameter=payload>\n{\"k\":[1,2]}\n</parameter>\n</function>";
        assert!(
            rt.accept_bytes(q_input),
            "qwen untyped structured value rejected"
        );
        assert!(rt.is_accepted());

        let mut rt2 = gemma4_runtime("f", schema);
        let g_input = b"call:f{payload:{k:[1,2]}}";
        assert!(
            rt2.accept_bytes(g_input),
            "gemma untyped structured value rejected"
        );
        assert!(rt2.is_accepted());
    }

    // -----------------------------------------------------------------------
    // iter-231c — regex `pattern` keyword → GBNF compilation
    // -----------------------------------------------------------------------

    /// The exact iter-231c driver: a tool schema whose array items carry
    /// `pattern: "^[a-z][a-z0-9-]*$"` (the ruvnet-brain `argv` shape that
    /// 400'd opencode).  Grammar must accept conforming items and reject
    /// non-conforming ones.
    #[test]
    fn iter231c_qwen35_pattern_on_array_items_enforced() {
        let schema = r#"{
            "type": "object",
            "properties": {
                "argv": {
                    "type": "array",
                    "items": {"type": "string", "pattern": "^[a-z][a-z0-9-]*$"}
                }
            }
        }"#;
        let mut rt = qwen35_runtime("cli_help", schema);
        let good = b"<function=cli_help>\n<parameter=argv>\n[\"ruflo\",\"claude-flow\",\"a\",\"z9-x\"]\n</parameter>\n</function>";
        assert!(rt.accept_bytes(good), "conforming argv rejected");
        assert!(rt.is_accepted());

        // "Bad" (uppercase start) violates the pattern — and the object
        // is a top-level parameter block, so there is no wildcard tail
        // to rescue it (top-level kleene alternation only).
        let mut rt2 = qwen35_runtime("cli_help", schema);
        let bad = b"<function=cli_help>\n<parameter=argv>\n[\"Bad\"]\n</parameter>\n</function>";
        let ok = rt2.accept_bytes(bad);
        assert!(
            !(ok && rt2.is_accepted()),
            "pattern-violating item accepted"
        );

        // "-x" (dash start) also violates it.
        let mut rt3 = qwen35_runtime("cli_help", schema);
        let bad2 = b"<function=cli_help>\n<parameter=argv>\n[\"-x\"]\n</parameter>\n</function>";
        assert!(
            !(rt3.accept_bytes(bad2) && rt3.is_accepted()),
            "dash-start item accepted"
        );
    }

    /// Quantified + alternation patterns on a nested string (year +
    /// http-method shapes).
    #[test]
    fn iter231c_qwen35_pattern_quantifier_and_alternation() {
        let schema = r#"{
            "type": "object",
            "properties": {
                "year": {"type": "string", "pattern": "^\\d{4}$"},
                "method": {"type": "string", "pattern": "^(get|post|delete)$"}
            },
            "required": ["year"]
        }"#;
        // NOTE: top-level Qwen STRING params render RAW (no quotes —
        // the template only `tojson`s non-string values), so the pattern
        // applies to the unquoted text.
        let mut rt = qwen35_runtime("f", schema);
        let good = b"<function=f>\n<parameter=year>\n2026\n</parameter>\n<parameter=method>\npost\n</parameter>\n</function>";
        assert!(
            rt.accept_bytes(good),
            "valid quantifier/alternation strings rejected"
        );
        assert!(rt.is_accepted());

        let mut rt2 = qwen35_runtime("f", schema);
        // "202" is 3 digits — violates \d{4}.
        let bad = b"<function=f>\n<parameter=year>\n202\n</parameter>\n</function>";
        assert!(
            !(rt2.accept_bytes(bad) && rt2.is_accepted()),
            "3-digit year accepted"
        );

        let mut rt3 = qwen35_runtime("f", schema);
        // "patch" is outside the alternation.
        let bad2 = b"<function=f>\n<parameter=year>\n2026\n</parameter>\n<parameter=method>\npatch\n</parameter>\n</function>";
        assert!(
            !(rt3.accept_bytes(bad2) && rt3.is_accepted()),
            "non-alternation method accepted"
        );
    }

    /// Unanchored pattern = JSON Schema "contains" semantics: prefix and
    /// suffix wildcards are added by the compiler.
    #[test]
    fn iter231c_qwen35_unanchored_pattern_is_contains() {
        let schema = r#"{
            "type": "object",
            "properties": {
                "text": {"type": "string", "pattern": "needle"}
            }
        }"#;
        let mut rt = qwen35_runtime("f", schema);
        let input = b"<function=f>\n<parameter=text>\n\"a haystack with needle inside\"\n</parameter>\n</function>";
        assert!(rt.accept_bytes(input), "contains-match rejected");
        assert!(rt.is_accepted());

        let mut rt2 = qwen35_runtime("f", schema);
        let bad = b"<function=f>\n<parameter=text>\n\"nothing here\"\n</parameter>\n</function>";
        assert!(
            !(rt2.accept_bytes(bad) && rt2.is_accepted()),
            "non-containing accepted"
        );
    }

    /// Gemma 4: pattern between the `<|"|>` markers.
    #[test]
    fn iter231c_gemma4_pattern_between_markers() {
        let schema = r#"{
            "type": "object",
            "properties": {
                "argv": {
                    "type": "array",
                    "items": {"type": "string", "pattern": "^[a-z][a-z0-9-]*$"}
                }
            }
        }"#;
        let mut rt = gemma4_runtime("cli_help", schema);
        let good = b"call:cli_help{argv:[<|\"|>ruflo<|\"|>,<|\"|>claude-flow<|\"|>]}";
        assert!(rt.accept_bytes(good), "conforming argv rejected (gemma)");
        assert!(rt.is_accepted());

        let mut rt2 = gemma4_runtime("cli_help", schema);
        let bad = b"call:cli_help{argv:[<|\"|>Bad<|\"|>]}";
        assert!(
            !(rt2.accept_bytes(bad) && rt2.is_accepted()),
            "pattern-violating item accepted (gemma)"
        );
    }

    /// Top-level string parameter with `pattern` (raw Qwen surface).
    #[test]
    fn iter231c_qwen35_toplevel_string_pattern() {
        let schema = r#"{
            "type": "object",
            "properties": {
                "subcommand": {"type": "string", "pattern": "^[a-z][a-z0-9-]*$"}
            }
        }"#;
        let mut rt = qwen35_runtime("cli", schema);
        let good = b"<function=cli>\n<parameter=subcommand>\nstatus\n</parameter>\n</function>";
        assert!(
            rt.accept_bytes(good),
            "valid top-level pattern string rejected"
        );
        assert!(rt.is_accepted());

        let mut rt2 = qwen35_runtime("cli", schema);
        let bad = b"<function=cli>\n<parameter=subcommand>\nStatus\n</parameter>\n</function>";
        assert!(
            !(rt2.accept_bytes(bad) && rt2.is_accepted()),
            "invalid top-level pattern string accepted"
        );
    }

    /// Non-regular regex features (backreference) remain an HONEST error
    /// — never a silent permissive downgrade.
    #[test]
    fn iter231c_nonregular_pattern_errors_honestly() {
        let schema = serde_json::json!({
            "type": "object",
            "properties": {
                "x": {"type": "string", "pattern": "^(a)\\1$"}
            }
        });
        let err = QWEN35
            .tool_call_gbnf("f", &schema, GrammarShape::SingleBody)
            .expect_err("backreference must error");
        assert!(
            err.contains("backreference"),
            "error names the feature: {}",
            err
        );
        assert!(err.contains("/x"), "error carries the path: {}", err);
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

    // -----------------------------------------------------------------------
    // Wave 2.7 W-η Q-A — eager-grammar acceptance + non-marker rejection
    //
    // Audit-driving tests for the `tool_choice=required` enforcement model.
    // These exercise the FULL emit→parse→runtime→mask path on real per-vocab
    // byte tables, mirroring the production decode loop's wiring.
    // -----------------------------------------------------------------------

    /// Helper: build a runtime for the eager (`OneOrMoreCalls`) shape.
    fn gemma4_required_runtime(
        fn_name: &str,
        schema_json: &str,
        parallel: bool,
    ) -> crate::serve::api::grammar::sampler::GrammarRuntime {
        let schema: serde_json::Value = serde_json::from_str(schema_json).unwrap();
        let gbnf = GEMMA4
            .tool_call_gbnf(fn_name, &schema, GrammarShape::OneOrMoreCalls { parallel })
            .unwrap_or_else(|e| panic!("tool_call_gbnf error: {}", e));
        grammar_runtime_for_gbnf(&gbnf)
    }

    fn qwen35_required_runtime(
        fn_name: &str,
        schema_json: &str,
        parallel: bool,
    ) -> crate::serve::api::grammar::sampler::GrammarRuntime {
        let schema: serde_json::Value = serde_json::from_str(schema_json).unwrap();
        let gbnf = QWEN35
            .tool_call_gbnf(fn_name, &schema, GrammarShape::OneOrMoreCalls { parallel })
            .unwrap_or_else(|e| panic!("tool_call_gbnf error: {}", e));
        grammar_runtime_for_gbnf(&gbnf)
    }

    /// T-QA-1: Gemma 4 eager grammar accepts the canonical marker-wrapped call.
    /// Validates the body bytes the model emits between markers (which the
    /// `ToolCallSplitter` swallows in production) by emitting the full
    /// `<|tool_call>...<tool_call|>` form at the runtime's grammar layer.
    #[test]
    fn gemma4_required_grammar_accepts_marker_wrapped_call() {
        let schema = r#"{
            "type": "object",
            "properties": {"location": {"type": "string"}}
        }"#;
        let mut rt = gemma4_required_runtime("get_weather", schema, false);
        let input = b"<|tool_call>call:get_weather{location:<|\"|>SF<|\"|>}<tool_call|>";
        assert!(
            rt.accept_bytes(input),
            "eager-grammar runtime must accept marker-wrapped call"
        );
        assert!(rt.is_accepted(), "not accepted at end");
    }

    /// T-QA-2 (audit-driving): mask path under `OneOrMoreCalls{parallel:false}`
    /// REJECTS the first non-marker byte at token 0.
    ///
    /// This is the structural-enforcement guarantee: the runtime is eager
    /// (`awaiting_trigger == false`) and the grammar root requires `<` (the
    /// first byte of `<|tool_call>`).  Any token whose decoded bytes don't
    /// prefix the open marker MUST be masked to -inf.
    ///
    /// Mirrors the peer's `grammar_lazy=false` for `tool_choice=REQUIRED`
    /// — the model is structurally unable
    /// to skip the tool call.
    #[test]
    fn required_eager_grammar_masks_non_marker_first_token() {
        use crate::serve::api::grammar::mask;
        let schema = r#"{
            "type": "object",
            "properties": {"location": {"type": "string"}}
        }"#;
        let rt = gemma4_required_runtime("get_weather", schema, false);
        assert!(
            !rt.is_awaiting_trigger(),
            "eager grammar runtime must NOT be in awaiting_trigger state"
        );
        // Build a tiny vocab: one token that starts the open marker (`<`),
        // and several non-marker tokens that the grammar must reject.
        // token_bytes table mirrors what `Engine::token_bytes_table`
        // produces from `Tokenizer::decode(&[id], false)`.
        let token_bytes: Vec<Vec<u8>> = vec![
            b"<".to_vec(),            // 0: starts <|tool_call> — survives
            b"<|tool_call>".to_vec(), // 1: full open marker     — survives
            b"hello".to_vec(),        // 2: plain content        — masked
            b"the".to_vec(),          // 3: plain content        — masked
            b" ".to_vec(),            // 4: whitespace           — masked
            b"\n".to_vec(),           // 5: newline              — masked
            b"call:".to_vec(),        // 6: body-prefix          — masked
            b"".to_vec(),             // 7: empty / EOS — illegal before acceptance
        ];
        let mut logits = vec![0.0_f32; token_bytes.len()];
        let masked = mask::mask_invalid_tokens(&rt, &token_bytes, &mut logits);

        // Marker-prefix tokens (0, 1) survive. Everything else, including an
        // empty/EOS piece before the grammar accepts, is masked.
        assert_eq!(masked, 6, "expected 6 masked tokens, logits = {:?}", logits);
        assert!(
            logits[0].is_finite(),
            "token 0 (`<`) must survive — prefixes open marker"
        );
        assert!(
            logits[1].is_finite(),
            "token 1 (`<|tool_call>`) must survive — full open marker"
        );
        assert!(!logits[2].is_finite(), "token 2 (`hello`) must be masked");
        assert!(!logits[3].is_finite(), "token 3 (`the`) must be masked");
        assert!(!logits[4].is_finite(), "token 4 (` `) must be masked");
        assert!(!logits[5].is_finite(), "token 5 (`\\n`) must be masked");
        assert!(
            !logits[6].is_finite(),
            "token 6 (`call:`) must be masked — body bytes only legal AFTER open marker"
        );
        assert!(
            !logits[7].is_finite(),
            "token 7 (empty bytes) must not bypass an unaccepted eager grammar"
        );
    }

    /// T-QA-3 (audit-driving): mask path under `SingleBody` + lazy
    /// (`awaiting_trigger=true`) ALLOWS every non-marker token at token 0.
    ///
    /// This is the AUTO-mode contract: the runtime is suspended until the
    /// `ToolCallSplitter` sees the open marker; preamble content tokens
    /// flow through unconstrained.  Matches the peer's `grammar_lazy=true`
    /// for `tool_choice=AUTO`.
    #[test]
    fn auto_lazy_grammar_allows_preamble_content() {
        use crate::serve::api::grammar::mask;
        let schema = r#"{
            "type": "object",
            "properties": {"location": {"type": "string"}}
        }"#;
        // Body-only grammar — what `compile_tool_grammar` would emit for AUTO
        // under a future Wave 2.7+ wiring.
        let mut rt = gemma4_runtime("get_weather", schema);
        // Arm the lazy gate the way `engine.rs` does for ToolCallBodyAuto.
        rt.set_awaiting_trigger(true);
        assert!(rt.is_awaiting_trigger());

        let token_bytes: Vec<Vec<u8>> = vec![
            b"<".to_vec(),
            b"<|tool_call>".to_vec(),
            b"hello".to_vec(),
            b"the".to_vec(),
            b" ".to_vec(),
            b"\n".to_vec(),
            b"call:".to_vec(),
        ];
        let mut logits = vec![0.0_f32; token_bytes.len()];
        let masked = mask::mask_invalid_tokens(&rt, &token_bytes, &mut logits);
        assert_eq!(
            masked, 0,
            "lazy/awaiting_trigger runtime must mask zero tokens"
        );
        for (i, l) in logits.iter().enumerate() {
            assert!(
                l.is_finite(),
                "token {} masked under awaiting_trigger; logits = {:?}",
                i,
                logits
            );
        }
    }

    /// T-QA-4: Qwen 3.5/3.6 eager grammar accepts the canonical
    /// marker-wrapped single call.
    #[test]
    fn qwen35_required_grammar_accepts_marker_wrapped_call() {
        let schema = r#"{
            "type": "object",
            "properties": {"location": {"type": "string"}}
        }"#;
        let mut rt = qwen35_required_runtime("get_weather", schema, false);
        // `<tool_call>\n<function=NAME>\n...\n</function>\n</tool_call>`.
        let input = b"<tool_call>\n<function=get_weather>\n<parameter=location>\nParis\n</parameter>\n</function>\n</tool_call>";
        assert!(
            rt.accept_bytes(input),
            "eager Qwen35 grammar must accept canonical single-call emission"
        );
        assert!(rt.is_accepted());
    }

    /// T-QA-5: under `parallel=false`, the eager grammar REJECTS a second
    /// open marker after the first call's close marker — single-call cap.
    #[test]
    fn gemma4_required_grammar_rejects_second_call_when_parallel_false() {
        let schema = r#"{
            "type": "object",
            "properties": {"location": {"type": "string"}}
        }"#;
        let mut rt = gemma4_required_runtime("get_weather", schema, false);
        // First call accepted in full.
        let first = b"<|tool_call>call:get_weather{location:<|\"|>SF<|\"|>}<tool_call|>";
        assert!(rt.accept_bytes(first));
        assert!(rt.is_accepted(), "first call must reach accepted state");
        // Second open marker must drive the runtime dead under parallel=false.
        let second_open = b"<|tool_call>";
        let alive = rt.accept_bytes(second_open);
        assert!(
            !alive || rt.is_dead(),
            "second `<|tool_call>` must be rejected under parallel=false"
        );
    }

    // -----------------------------------------------------------------------
    // Wave 2.7 W-η Q-B — parallel_tool_calls grammar shape
    //
    // Audit-driving tests for the multi-call grammar shape per Codex audit
    // divergence "Multi-tool re-entry" (severity med): the wave-2.6 worker
    // landed the no-reset-on-close runtime semantics but kept the single-call
    // grammar shape, so under `parallel_tool_calls=true` the model emitting a
    // second open marker would silently drive the runtime dead.  Q-B emits
    // `(call)+` (Gemma) and `call ("\n" call)*` (Qwen) so the runtime
    // structurally accepts the natural multi-call template output.
    // -----------------------------------------------------------------------

    /// T-QB-1 (audit-driving): Gemma 4 parallel grammar accepts two
    /// back-to-back calls with no separator (chat_template.jinja:189-205).
    #[test]
    fn gemma4_parallel_grammar_accepts_two_calls() {
        let schema = r#"{
            "type": "object",
            "properties": {"location": {"type": "string"}}
        }"#;
        let mut rt = gemma4_required_runtime("get_weather", schema, true);
        // Two back-to-back calls — Gemma chat_template emits with NO separator.
        let input = b"<|tool_call>call:get_weather{location:<|\"|>SF<|\"|>}<tool_call|>\
                     <|tool_call>call:get_weather{location:<|\"|>NYC<|\"|>}<tool_call|>";
        assert!(
            rt.accept_bytes(input),
            "parallel Gemma 4 grammar must accept two back-to-back calls"
        );
        assert!(rt.is_accepted(), "two-call sequence not in accepting state");
    }

    /// T-QB-2 (audit-driving): Qwen 3.5/3.6 parallel grammar accepts two
    /// calls separated by a literal `\n` (tokenizer_config.json:285+).
    #[test]
    fn qwen35_parallel_grammar_accepts_two_calls_separated_by_newline() {
        let schema = r#"{
            "type": "object",
            "properties": {"location": {"type": "string"}}
        }"#;
        let mut rt = qwen35_required_runtime("get_weather", schema, true);
        // Two calls separated by literal `\n` per chat_template's `else` branch.
        let input = b"<tool_call>\n<function=get_weather>\n<parameter=location>\nParis\n</parameter>\n</function>\n</tool_call>\n<tool_call>\n<function=get_weather>\n<parameter=location>\nLondon\n</parameter>\n</function>\n</tool_call>";
        assert!(
            rt.accept_bytes(input),
            "parallel Qwen35 grammar must accept two calls with `\\n` separator"
        );
        assert!(rt.is_accepted(), "two-call sequence not accepted");
    }

    /// T-QB-3 (audit-driving): Qwen 3.5/3.6 parallel grammar REJECTS two
    /// calls without the `\n` separator — the chat_template requires the
    /// newline, so the grammar must too.
    #[test]
    fn qwen35_parallel_grammar_rejects_calls_without_newline_separator() {
        let schema = r#"{
            "type": "object",
            "properties": {"location": {"type": "string"}}
        }"#;
        let mut rt = qwen35_required_runtime("get_weather", schema, true);
        // Two calls back-to-back with NO separator.  This violates the Qwen
        // chat-template contract; the grammar must reject it.
        let input = b"<tool_call>\n<function=get_weather>\n<parameter=location>\nParis\n</parameter>\n</function>\n</tool_call><tool_call>\n<function=get_weather>\n<parameter=location>\nLondon\n</parameter>\n</function>\n</tool_call>";
        let alive = rt.accept_bytes(input);
        assert!(
            !alive || !rt.is_accepted(),
            "Qwen parallel grammar accepted two calls without `\\n` separator (should reject)"
        );
    }

    /// T-QB-4: under `parallel = false`, Qwen 3.5/3.6 grammar REJECTS the
    /// trailing `\n<tool_call>` of a second call.  Mirrors the Gemma 4
    /// equivalent for Qwen's separator semantics.
    #[test]
    fn qwen35_required_grammar_rejects_second_call_when_parallel_false() {
        let schema = r#"{
            "type": "object",
            "properties": {"location": {"type": "string"}}
        }"#;
        let mut rt = qwen35_required_runtime("get_weather", schema, false);
        let first = b"<tool_call>\n<function=get_weather>\n<parameter=location>\nParis\n</parameter>\n</function>\n</tool_call>";
        assert!(rt.accept_bytes(first));
        assert!(rt.is_accepted(), "first call must reach accepted state");
        let second = b"\n<tool_call>";
        let alive = rt.accept_bytes(second);
        assert!(
            !alive || rt.is_dead(),
            "second call open must be rejected under parallel=false"
        );
    }
}
