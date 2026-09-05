//! XGrammar structural-tag JSON lowered into the shared GBNF runtime.
//!
//! The accepted surface is intentionally closed: a malformed or token-aware
//! format is an error, never unconstrained output.  String-triggered tags are
//! compiled as a finite-state scanner so free text cannot consume a trigger.

use std::collections::{BTreeMap, BTreeSet};

use serde_json::{Map, Value};

use super::{json_schema, parser, regex_gbnf, serialize, Grammar};

const MAX_CHAR_BOUND: u64 = 2_000;
const MAX_DISPATCH_STATES: usize = 4_096;

/// Compile either the current XGrammar object or vLLM's legacy structure.
pub fn compile(payload: &Value) -> Result<Grammar, StructuralTagError> {
    let source = lower_to_gbnf(payload)?;
    parser::parse_generated(&source).map_err(|error| {
        StructuralTagError::Invalid(format!("generated GBNF was invalid: {error}"))
    })
}

/// Compile a structural tag with a tokenizer-bound token-string resolver.
///
/// Numeric token IDs work with [`compile`]. XGrammar also accepts tokenizer
/// token strings, which must resolve to exactly one authoritative token ID at
/// the serving boundary; this API keeps that model dependency explicit.
pub fn compile_with_token_resolver<F>(
    payload: &Value,
    resolver: F,
) -> Result<Grammar, StructuralTagError>
where
    F: FnMut(&str) -> Result<u32, String>,
{
    let source = lower_to_gbnf_with_token_resolver(payload, resolver)?;
    parser::parse_generated(&source).map_err(|error| {
        StructuralTagError::Invalid(format!("generated GBNF was invalid: {error}"))
    })
}

/// Lower either accepted structural-tag shape to standalone GBNF text.
pub fn lower_to_gbnf(payload: &Value) -> Result<String, StructuralTagError> {
    lower_to_gbnf_inner(payload, None)
}

/// Lower a structural tag while resolving XGrammar token-string values.
pub fn lower_to_gbnf_with_token_resolver<F>(
    payload: &Value,
    mut resolver: F,
) -> Result<String, StructuralTagError>
where
    F: FnMut(&str) -> Result<u32, String>,
{
    lower_to_gbnf_inner(payload, Some(&mut resolver))
}

fn lower_to_gbnf_inner(
    payload: &Value,
    resolver: Option<&mut dyn FnMut(&str) -> Result<u32, String>>,
) -> Result<String, StructuralTagError> {
    let root = object(payload, "structural_tag")?;
    exact_keys(
        root,
        &["type", "format", "structures", "triggers"],
        "structural_tag",
    )?;
    expect_type(root, "structural_tag", "structural_tag")?;
    let has_format = root.contains_key("format");
    let has_legacy = root.contains_key("structures") || root.contains_key("triggers");
    if has_format && has_legacy {
        return Err(StructuralTagError::Invalid(
            "structural_tag may use current format or legacy structures/triggers, not both".into(),
        ));
    }
    let mut lowerer = Lowerer::new(resolver);
    let entry = if has_format {
        lowerer.format(required(root, "format", "structural_tag")?)?
    } else if has_legacy {
        let structures = array(
            required(root, "structures", "structural_tag")?,
            "structures",
        )?;
        let triggers = strings(required(root, "triggers", "structural_tag")?, "triggers")?;
        lowerer.legacy(structures, &triggers)?
    } else {
        return Err(StructuralTagError::Invalid(
            "structural_tag requires format or structures plus triggers".into(),
        ));
    };
    Ok(lowerer.finish(&entry))
}

struct Lowerer<'a> {
    next: usize,
    rules: Vec<(String, String)>,
    token_resolver: Option<&'a mut dyn FnMut(&str) -> Result<u32, String>>,
}

impl<'a> Lowerer<'a> {
    fn new(token_resolver: Option<&'a mut dyn FnMut(&str) -> Result<u32, String>>) -> Self {
        Self {
            next: 0,
            rules: Vec::new(),
            token_resolver,
        }
    }
    fn rule(&mut self, body: impl Into<String>) -> String {
        let name = format!("struct-{}", self.next);
        self.next += 1;
        self.rules.push((name.clone(), body.into()));
        name
    }

    fn finish(self, entry: &str) -> String {
        let mut source = format!("root ::= {entry}\n");
        for (name, body) in self.rules {
            source.push_str(&format!("{name} ::= {body}\n"));
        }
        source
    }

    fn embed(&mut self, source: &str, context: &str) -> Result<String, StructuralTagError> {
        let grammar = parser::parse(source).map_err(|error| {
            StructuralTagError::Invalid(format!("{context} is not valid GBNF: {error}"))
        })?;
        if grammar.rule_id("root").is_none() {
            return Err(StructuralTagError::Invalid(format!(
                "{context} must define a root rule"
            )));
        }
        let prefix = format!("embedded-{}-", self.next);
        self.next += 1;
        let renamed = serialize::rename_rules(&grammar, |name| format!("{prefix}{name}"));
        self.rules
            .extend(split_rules(&serialize::serialize(&renamed))?);
        Ok(format!("{prefix}root"))
    }

    fn format(&mut self, value: &Value) -> Result<String, StructuralTagError> {
        self.format_with_token_end(value, None)
    }

    fn format_with_token_end(
        &mut self,
        value: &Value,
        enclosing_end_token: Option<u32>,
    ) -> Result<String, StructuralTagError> {
        let map = object(value, "format")?;
        let kind = string(required(map, "type", "format")?, "format.type")?;
        match kind {
            "const_string" => {
                exact_keys(map, &["type", "value"], kind)?;
                Ok(self.rule(literal(string(required(map, "value", kind)?, "const_string.value")?)?))
            }
            "json_schema" => self.json_schema(map),
            "grammar" => {
                exact_keys(map, &["type", "grammar"], kind)?;
                self.embed(string(required(map, "grammar", kind)?, "grammar.grammar")?, "structural_tag grammar")
            }
            "regex" => {
                exact_keys(map, &["type", "pattern"], kind)?;
                let body = regex_gbnf::regex_to_gbnf_full_match(
                    string(required(map, "pattern", kind)?, "regex.pattern")?,
                    regex_gbnf::Surface::RawOutput,
                )
                .map_err(|error| StructuralTagError::Invalid(format!("invalid structural-tag regex: {error}")))?;
                Ok(self.rule(body))
            }
            "any_text" => self.any_text(map),
            "token" => self.token(map),
            "exclude_token" => self.exclude_token(map),
            "any_tokens" => self.any_tokens(map, enclosing_end_token),
            "sequence" | "or" => self.list(map, kind),
            "optional" | "plus" | "star" => self.unary(map, kind),
            "repeat" => self.repeat(map),
            "tag" => self.tag(map),
            "triggered_tags" => self.triggered(map),
            "tags_with_separator" => self.tags_with_separator(map),
            "dispatch" => self.dispatch(map),
            "token_triggered_tags" => self.token_triggered(map),
            "token_dispatch" => self.token_dispatch(map),
            "qwen_xml_parameter" => Err(StructuralTagError::Unsupported("qwen_xml_parameter requires an XML-schema lowering not provided by the JSON-only foundation".into())),
            _ => Err(StructuralTagError::Invalid(format!("unknown structural-tag format type '{kind}'"))),
        }
    }

    fn json_schema(&mut self, map: &Map<String, Value>) -> Result<String, StructuralTagError> {
        exact_keys(
            map,
            &[
                "type",
                "json_schema",
                "style",
                "any_order",
                "max_whitespace_cnt",
            ],
            "json_schema",
        )?;
        if let Some(style) = map.get("style") {
            let style = string(style, "json_schema.style")?;
            if style != "json" {
                return Err(StructuralTagError::Unsupported(format!(
                    "json_schema style '{style}' requires model-specific XML lowering"
                )));
            }
        }
        let any_order = optional_bool(map, "any_order", false, "json_schema")?;
        let whitespace = match map.get("max_whitespace_cnt") {
            None | Some(Value::Null) => None,
            Some(value) => {
                let maximum = bounded(value, "json_schema.max_whitespace_cnt")?;
                if maximum == 0 {
                    return Err(StructuralTagError::Invalid(
                        "json_schema.max_whitespace_cnt must be positive when present".into(),
                    ));
                }
                Some(format!("[ \\n\\r\\t]{{0,{maximum}}}"))
            }
        };
        let schema = required(map, "json_schema", "json_schema")?;
        if !schema.is_object() && !schema.is_boolean() {
            return Err(StructuralTagError::Invalid(
                "json_schema.json_schema must be an object or boolean".into(),
            ));
        }
        let source = json_schema::schema_to_gbnf_for_structural_tag(
            schema,
            whitespace.as_deref(),
            any_order,
        )
        .map_err(|error| {
            StructuralTagError::Invalid(format!("invalid structural-tag JSON Schema: {error}"))
        })?;
        self.embed(&source, "structural-tag JSON Schema")
    }

    fn token(&mut self, map: &Map<String, Value>) -> Result<String, StructuralTagError> {
        exact_keys(map, &["type", "token"], "token")?;
        let token = self.resolve_token(required(map, "token", "token")?, "token.token")?;
        Ok(self.rule(token_expression(token)?))
    }

    fn exclude_token(&mut self, map: &Map<String, Value>) -> Result<String, StructuralTagError> {
        exact_keys(map, &["type", "exclude_tokens"], "exclude_token")?;
        let excludes = map
            .get("exclude_tokens")
            .map(|value| self.resolve_tokens(value, "exclude_token.exclude_tokens"))
            .transpose()?
            .unwrap_or_default();
        Ok(self.rule(exclude_tokens_expression(&excludes)?))
    }

    fn any_tokens(
        &mut self,
        map: &Map<String, Value>,
        enclosing_end_token: Option<u32>,
    ) -> Result<String, StructuralTagError> {
        exact_keys(map, &["type", "exclude_tokens", "max_tokens"], "any_tokens")?;
        let mut excludes = map
            .get("exclude_tokens")
            .map(|value| self.resolve_tokens(value, "any_tokens.exclude_tokens"))
            .transpose()?
            .unwrap_or_default();
        if let Some(end) = enclosing_end_token {
            excludes.push(end);
        }
        canonicalize_tokens(&mut excludes);
        let max = match map.get("max_tokens") {
            None | Some(Value::Null) => None,
            Some(value) => Some(bounded(value, "any_tokens.max_tokens")?),
        };
        let atom = exclude_tokens_expression(&excludes)?;
        let body = match max {
            None => format!("( {atom} )*"),
            Some(0) => "\"\"".to_owned(),
            Some(maximum) => format!("( {atom} ){{0,{maximum}}}"),
        };
        Ok(self.rule(body))
    }

    fn resolve_token(&mut self, value: &Value, context: &str) -> Result<u32, StructuralTagError> {
        if let Some(id) = value.as_u64() {
            return u32::try_from(id).map_err(|_| {
                StructuralTagError::Invalid(format!("{context} token id is outside the u32 range"))
            });
        }
        let token = string(value, context)?;
        let Some(resolver) = self.token_resolver.as_mut() else {
            return Err(StructuralTagError::NeedsTokenVocabulary(context.to_owned()));
        };
        resolver(token).map_err(|error| {
            StructuralTagError::Invalid(format!(
                "{context} token {token:?} did not resolve: {error}"
            ))
        })
    }

    fn resolve_tokens(
        &mut self,
        value: &Value,
        context: &str,
    ) -> Result<Vec<u32>, StructuralTagError> {
        let mut tokens = Vec::new();
        for (index, value) in array(value, context)?.iter().enumerate() {
            tokens.push(self.resolve_token(value, &format!("{context}[{index}]"))?);
        }
        canonicalize_tokens(&mut tokens);
        Ok(tokens)
    }

    fn terminal(&mut self, value: &Value, context: &str) -> Result<Terminal, StructuralTagError> {
        match value {
            Value::String(value) => Ok(Terminal::Text(value.clone())),
            Value::Object(map) => {
                exact_keys(map, &["type", "token"], context)?;
                expect_type(map, "token", context)?;
                Ok(Terminal::Token(self.resolve_token(
                    required(map, "token", context)?,
                    &format!("{context}.token"),
                )?))
            }
            _ => Err(StructuralTagError::Invalid(format!(
                "{context} must be a string or token format"
            ))),
        }
    }

    fn terminals(
        &mut self,
        value: &Value,
        context: &str,
    ) -> Result<Vec<Terminal>, StructuralTagError> {
        match value {
            Value::Array(values) => values
                .iter()
                .enumerate()
                .map(|(index, value)| self.terminal(value, &format!("{context}[{index}]")))
                .collect(),
            _ => Ok(vec![self.terminal(value, context)?]),
        }
    }

    fn single_terminal(
        &mut self,
        value: &Value,
        context: &str,
    ) -> Result<Terminal, StructuralTagError> {
        let mut terminals = self.terminals(value, context)?;
        if terminals.len() != 1 {
            return Err(StructuralTagError::Unsupported(format!(
                "{context} must contain exactly one terminal"
            )));
        }
        Ok(terminals.remove(0))
    }

    fn any_text(&mut self, map: &Map<String, Value>) -> Result<String, StructuralTagError> {
        exact_keys(
            map,
            &["type", "excludes", "max_tokens", "max_chars"],
            "any_text",
        )?;
        let excludes = map
            .get("excludes")
            .map(|value| strings(value, "any_text.excludes"))
            .transpose()?
            .unwrap_or_default();
        let max_tokens = match map.get("max_tokens") {
            None | Some(Value::Null) => None,
            Some(value) => Some(bounded(value, "any_text.max_tokens")?),
        };
        if let Some(maximum) = max_tokens {
            if !excludes.is_empty() {
                return Err(StructuralTagError::Unsupported(
                    "any_text cannot combine max_tokens with string excludes in the token-terminal runtime"
                        .into(),
                ));
            }
            let body = if maximum == 0 {
                "\"\"".to_owned()
            } else {
                format!("( <[*]> ){{0,{maximum}}}")
            };
            return Ok(self.rule(body));
        }
        let max_chars = match map.get("max_chars") {
            None | Some(Value::Null) => None,
            Some(value) => Some(bounded(value, "any_text.max_chars")?),
        };
        if excludes.is_empty() {
            let body = match max_chars {
                None => "[^\\x00]*".to_owned(),
                Some(maximum) => format!("[^\\x00]{{0,{maximum}}}"),
            };
            return Ok(self.rule(body));
        }
        validate_scan_patterns(&[], &excludes)?;
        let entry = {
            let mut scanner = Scanner::new(self, Vec::new(), &[], &excludes, false, max_chars)?;
            scanner.start(false)?
        };
        Ok(self.rule(entry))
    }

    fn list(&mut self, map: &Map<String, Value>, kind: &str) -> Result<String, StructuralTagError> {
        exact_keys(map, &["type", "elements"], kind)?;
        let elements = array(
            required(map, "elements", kind)?,
            &format!("{kind}.elements"),
        )?;
        if kind == "or" && elements.is_empty() {
            return Err(StructuralTagError::Invalid(
                "or.elements must not be empty".into(),
            ));
        }
        let mut lowered = Vec::with_capacity(elements.len());
        for element in elements {
            lowered.push(self.format(element)?);
        }
        let body = if lowered.is_empty() {
            "\"\"".to_owned()
        } else if kind == "or" {
            format!("( {} )", lowered.join(" | "))
        } else {
            lowered.join(" ")
        };
        Ok(self.rule(body))
    }

    fn unary(
        &mut self,
        map: &Map<String, Value>,
        kind: &str,
    ) -> Result<String, StructuralTagError> {
        exact_keys(map, &["type", "content"], kind)?;
        let content = self.format(required(map, "content", kind)?)?;
        let suffix = match kind {
            "optional" => "?",
            "plus" => "+",
            "star" => "*",
            _ => unreachable!(),
        };
        Ok(self.rule(format!("{content}{suffix}")))
    }

    fn repeat(&mut self, map: &Map<String, Value>) -> Result<String, StructuralTagError> {
        exact_keys(map, &["type", "min", "max", "content"], "repeat")?;
        let min = bounded(required(map, "min", "repeat")?, "repeat.min")?;
        let max = integer(required(map, "max", "repeat")?, "repeat.max")?;
        if max < -1 || (max >= 0 && (max as u64) < min) {
            return Err(StructuralTagError::Invalid(
                "repeat.max must be -1 or >= repeat.min".into(),
            ));
        }
        if max > MAX_CHAR_BOUND as i64 {
            return Err(StructuralTagError::Invalid(format!(
                "repeat.max exceeds GBNF limit {MAX_CHAR_BOUND}"
            )));
        }
        let content = self.format(required(map, "content", "repeat")?)?;
        let quantifier = if max == -1 {
            format!("{{{min},}}")
        } else {
            format!("{{{min},{max}}}")
        };
        Ok(self.rule(format!("{content}{quantifier}")))
    }

    fn tag(&mut self, map: &Map<String, Value>) -> Result<String, StructuralTagError> {
        exact_keys(map, &["type", "begin", "content", "end"], "tag")?;
        let begin = self.terminal(required(map, "begin", "tag")?, "tag.begin")?;
        let end = self.terminals(required(map, "end", "tag")?, "tag.end")?;
        let token_end = (end.len() == 1).then(|| end[0].token_id()).flatten();
        let content = self.format_with_token_end(required(map, "content", "tag")?, token_end)?;
        Ok(self.rule(format!(
            "{} {content} {}",
            begin.expression()?,
            terminal_expression(&end)?
        )))
    }

    fn tags_with_separator(
        &mut self,
        map: &Map<String, Value>,
    ) -> Result<String, StructuralTagError> {
        exact_keys(
            map,
            &[
                "type",
                "tags",
                "separator",
                "at_least_one",
                "stop_after_first",
            ],
            "tags_with_separator",
        )?;
        let tags = array(
            required(map, "tags", "tags_with_separator")?,
            "tags_with_separator.tags",
        )?;
        if tags.is_empty() {
            return Err(StructuralTagError::Invalid(
                "tags_with_separator.tags must not be empty".into(),
            ));
        }
        let mut rules = Vec::new();
        for tag in tags {
            rules.push(self.tag(object(tag, "tags_with_separator.tags item")?)?);
        }
        let tags = if rules.len() == 1 {
            rules.remove(0)
        } else {
            self.rule(format!("( {} )", rules.join(" | ")))
        };
        let separator = literal(string(
            required(map, "separator", "tags_with_separator")?,
            "tags_with_separator.separator",
        )?)?;
        let required = optional_bool(map, "at_least_one", false, "tags_with_separator")?;
        let stop = optional_bool(map, "stop_after_first", false, "tags_with_separator")?;
        let body = if stop {
            if required {
                tags
            } else {
                format!("{tags}?")
            }
        } else if required {
            format!("{tags} ({separator} {tags})*")
        } else {
            format!("({tags} ({separator} {tags})*)?")
        };
        Ok(self.rule(body))
    }

    fn legacy(
        &mut self,
        structures: &[Value],
        triggers: &[String],
    ) -> Result<String, StructuralTagError> {
        if structures.is_empty() || triggers.is_empty() {
            return Err(StructuralTagError::Invalid(
                "legacy structures and triggers must not be empty".into(),
            ));
        }
        let mut tags = Vec::new();
        for structure in structures {
            let map = object(structure, "legacy structure")?;
            exact_keys(map, &["begin", "schema", "end"], "legacy structure")?;
            let schema = required(map, "schema", "legacy structure")?;
            if !schema.is_object() && !schema.is_boolean() {
                return Err(StructuralTagError::Invalid(
                    "legacy structure.schema must be a JSON Schema object or boolean".into(),
                ));
            }
            let content = self.embed(
                &json_schema::schema_to_gbnf(schema).map_err(|e| {
                    StructuralTagError::Invalid(format!("invalid legacy JSON Schema: {e}"))
                })?,
                "legacy JSON Schema",
            )?;
            tags.push(TagSpec {
                begin: Terminal::Text(
                    string(
                        required(map, "begin", "legacy structure")?,
                        "legacy structure.begin",
                    )?
                    .to_owned(),
                ),
                content,
                end: vec![Terminal::Text(
                    string(
                        required(map, "end", "legacy structure")?,
                        "legacy structure.end",
                    )?
                    .to_owned(),
                )],
            });
        }
        self.string_triggered(tags, triggers, &[], false, false)
    }

    fn triggered(&mut self, map: &Map<String, Value>) -> Result<String, StructuralTagError> {
        exact_keys(
            map,
            &[
                "type",
                "triggers",
                "tags",
                "at_least_one",
                "stop_after_first",
                "excludes",
            ],
            "triggered_tags",
        )?;
        let excludes = map
            .get("excludes")
            .map(|value| strings(value, "triggered_tags.excludes"))
            .transpose()?
            .unwrap_or_default();
        let triggers = strings(
            required(map, "triggers", "triggered_tags")?,
            "triggered_tags.triggers",
        )?;
        let mut tags = Vec::new();
        for tag in array(
            required(map, "tags", "triggered_tags")?,
            "triggered_tags.tags",
        )? {
            let tag = object(tag, "triggered_tags tag")?;
            exact_keys(
                tag,
                &["type", "begin", "content", "end"],
                "triggered_tags tag",
            )?;
            expect_type(tag, "tag", "triggered_tags tag")?;
            tags.push(TagSpec {
                begin: match self.terminal(
                    required(tag, "begin", "triggered_tags tag")?,
                    "triggered_tags tag.begin",
                )? {
                    Terminal::Text(text) => Terminal::Text(text),
                    Terminal::Token(_) => {
                        return Err(StructuralTagError::Invalid(
                            "triggered_tags tag.begin must be a string; use token_triggered_tags"
                                .into(),
                        ));
                    }
                },
                content: self.format(required(tag, "content", "triggered_tags tag")?)?,
                end: self.terminals(
                    required(tag, "end", "triggered_tags tag")?,
                    "triggered_tags tag.end",
                )?,
            });
        }
        self.string_triggered(
            tags,
            &triggers,
            &excludes,
            optional_bool(map, "at_least_one", false, "triggered_tags")?,
            optional_bool(map, "stop_after_first", false, "triggered_tags")?,
        )
    }

    fn dispatch(&mut self, map: &Map<String, Value>) -> Result<String, StructuralTagError> {
        exact_keys(map, &["type", "rules", "loop", "excludes"], "dispatch")?;
        let rules = array(required(map, "rules", "dispatch")?, "dispatch.rules")?;
        if rules.is_empty() {
            return Err(StructuralTagError::Invalid(
                "dispatch.rules must not be empty".into(),
            ));
        }
        let mut triggers = Vec::with_capacity(rules.len());
        let mut tags = Vec::with_capacity(rules.len());
        for (index, rule) in rules.iter().enumerate() {
            let pair = array(rule, &format!("dispatch.rules/{index}"))?;
            if pair.len() != 2 {
                return Err(StructuralTagError::Invalid(format!(
                    "dispatch.rules/{index} must contain [pattern, format]"
                )));
            }
            let pattern = string(&pair[0], &format!("dispatch.rules/{index}/0"))?.to_owned();
            let content = self.format(&pair[1])?;
            triggers.push(pattern.clone());
            tags.push(TagSpec {
                begin: Terminal::Text(pattern),
                content,
                end: vec![Terminal::Text(String::new())],
            });
        }
        let excludes = map
            .get("excludes")
            .map(|value| strings(value, "dispatch.excludes"))
            .transpose()?
            .unwrap_or_default();
        self.string_triggered(
            tags,
            &triggers,
            &excludes,
            false,
            !optional_bool(map, "loop", true, "dispatch")?,
        )
    }

    fn token_triggered(&mut self, map: &Map<String, Value>) -> Result<String, StructuralTagError> {
        exact_keys(
            map,
            &[
                "type",
                "trigger_tokens",
                "tags",
                "exclude_tokens",
                "at_least_one",
                "stop_after_first",
            ],
            "token_triggered_tags",
        )?;
        let triggers = self.resolve_tokens(
            required(map, "trigger_tokens", "token_triggered_tags")?,
            "token_triggered_tags.trigger_tokens",
        )?;
        if triggers.is_empty() {
            return Err(StructuralTagError::Invalid(
                "token_triggered_tags.trigger_tokens must not be empty".into(),
            ));
        }
        let excludes = map
            .get("exclude_tokens")
            .map(|value| self.resolve_tokens(value, "token_triggered_tags.exclude_tokens"))
            .transpose()?
            .unwrap_or_default();
        let mut tags = Vec::new();
        for (index, value) in array(
            required(map, "tags", "token_triggered_tags")?,
            "token_triggered_tags.tags",
        )?
        .iter()
        .enumerate()
        {
            let tag = object(value, &format!("token_triggered_tags.tags[{index}]"))?;
            exact_keys(
                tag,
                &["type", "begin", "content", "end"],
                "token_triggered_tags tag",
            )?;
            expect_type(tag, "tag", "token_triggered_tags tag")?;
            let begin = self.terminal(
                required(tag, "begin", "token_triggered_tags tag")?,
                "token_triggered_tags tag.begin",
            )?;
            let Some(begin_token) = begin.token_id() else {
                return Err(StructuralTagError::Invalid(
                    "token_triggered_tags tag.begin must be a token format".into(),
                ));
            };
            if !triggers.contains(&begin_token) {
                return Err(StructuralTagError::Invalid(format!(
                    "token-triggered tag begin token {begin_token} is absent from trigger_tokens"
                )));
            }
            let end = self.terminals(
                required(tag, "end", "token_triggered_tags tag")?,
                "token_triggered_tags tag.end",
            )?;
            let end_token = (end.len() == 1).then(|| end[0].token_id()).flatten();
            let content = self.format_with_token_end(
                required(tag, "content", "token_triggered_tags tag")?,
                end_token,
            )?;
            tags.push(TagSpec {
                begin,
                content,
                end,
            });
        }
        self.token_triggered_inner(
            tags,
            &triggers,
            &excludes,
            optional_bool(map, "at_least_one", false, "token_triggered_tags")?,
            optional_bool(map, "stop_after_first", false, "token_triggered_tags")?,
        )
    }

    fn token_dispatch(&mut self, map: &Map<String, Value>) -> Result<String, StructuralTagError> {
        exact_keys(
            map,
            &["type", "rules", "loop", "exclude_tokens"],
            "token_dispatch",
        )?;
        let rules = array(
            required(map, "rules", "token_dispatch")?,
            "token_dispatch.rules",
        )?;
        if rules.is_empty() {
            return Err(StructuralTagError::Invalid(
                "token_dispatch.rules must not be empty".into(),
            ));
        }
        let mut triggers = Vec::with_capacity(rules.len());
        let mut tags = Vec::with_capacity(rules.len());
        for (index, rule) in rules.iter().enumerate() {
            let pair = array(rule, &format!("token_dispatch.rules[{index}]"))?;
            if pair.len() != 2 {
                return Err(StructuralTagError::Invalid(format!(
                    "token_dispatch.rules[{index}] must contain [token, format]"
                )));
            }
            let token =
                self.resolve_token(&pair[0], &format!("token_dispatch.rules[{index}][0]"))?;
            triggers.push(token);
            tags.push(TagSpec {
                begin: Terminal::Token(token),
                content: self.format(&pair[1])?,
                end: vec![Terminal::Text(String::new())],
            });
        }
        canonicalize_tokens(&mut triggers);
        let excludes = map
            .get("exclude_tokens")
            .map(|value| self.resolve_tokens(value, "token_dispatch.exclude_tokens"))
            .transpose()?
            .unwrap_or_default();
        self.token_triggered_inner(
            tags,
            &triggers,
            &excludes,
            false,
            !optional_bool(map, "loop", true, "token_dispatch")?,
        )
    }

    fn token_triggered_inner(
        &mut self,
        tags: Vec<TagSpec>,
        triggers: &[u32],
        excludes: &[u32],
        at_least_one: bool,
        stop_after_first: bool,
    ) -> Result<String, StructuralTagError> {
        if tags.is_empty() {
            return Err(StructuralTagError::Invalid(
                "token-triggered tags must not be empty".into(),
            ));
        }
        let state = self.rule("");
        let tail = if stop_after_first {
            "\"\"".to_owned()
        } else {
            state.clone()
        };
        let mut tagged = Vec::new();
        for tag in &tags {
            for end in &tag.end {
                tagged.push(format!(
                    "{} {} {} {}",
                    tag.begin.expression()?,
                    tag.content,
                    end.expression()?,
                    tail
                ));
            }
        }
        let mut blocked = triggers.to_vec();
        blocked.extend_from_slice(excludes);
        canonicalize_tokens(&mut blocked);
        let free = exclude_tokens_expression(&blocked)?;
        let body = format!("( {} | {free} {state} | \"\" )", tagged.join(" | "));
        let rule = self
            .rules
            .iter_mut()
            .find(|(name, _)| name == &state)
            .expect("token dispatch state rule exists");
        rule.1 = body;
        if at_least_one {
            Ok(self.rule(format!("( {} )", tagged.join(" | "))))
        } else {
            Ok(state)
        }
    }

    fn string_triggered(
        &mut self,
        tags: Vec<TagSpec>,
        triggers: &[String],
        excludes: &[String],
        at_least_one: bool,
        stop_after_first: bool,
    ) -> Result<String, StructuralTagError> {
        validate_trigger_topology(&tags, triggers)?;
        validate_scan_patterns(triggers, excludes)?;
        let mut scanner = Scanner::new(self, tags, triggers, excludes, stop_after_first, None)?;
        let required_start = scanner.start(at_least_one)?;
        Ok(self.rule(required_start))
    }
}

#[derive(Clone, Debug)]
enum Terminal {
    Text(String),
    Token(u32),
}

impl Terminal {
    fn expression(&self) -> Result<String, StructuralTagError> {
        match self {
            Self::Text(value) => literal(value),
            Self::Token(token) => token_expression(*token),
        }
    }

    fn as_text(&self) -> Option<&String> {
        match self {
            Self::Text(value) => Some(value),
            Self::Token(_) => None,
        }
    }

    fn token_id(&self) -> Option<u32> {
        match self {
            Self::Text(_) => None,
            Self::Token(token) => Some(*token),
        }
    }
}

#[derive(Clone)]
struct TagSpec {
    begin: Terminal,
    content: String,
    end: Vec<Terminal>,
}

struct Scanner<'lowerer, 'resolver> {
    lowerer: &'lowerer mut Lowerer<'resolver>,
    tags: Vec<TagSpec>,
    triggers: Vec<String>,
    excludes: Vec<String>,
    stop: bool,
    max_chars: Option<u64>,
    states: BTreeMap<(String, Option<u64>), String>,
}

impl<'lowerer, 'resolver> Scanner<'lowerer, 'resolver> {
    fn new(
        lowerer: &'lowerer mut Lowerer<'resolver>,
        tags: Vec<TagSpec>,
        triggers: &[String],
        excludes: &[String],
        stop: bool,
        max_chars: Option<u64>,
    ) -> Result<Self, StructuralTagError> {
        let state_count = triggers
            .iter()
            .chain(excludes)
            .map(|pattern| pattern.chars().count().max(1))
            .sum::<usize>()
            .max(1)
            .saturating_mul(max_chars.map_or(1, |maximum| maximum as usize + 1));
        if state_count > MAX_DISPATCH_STATES {
            return Err(StructuralTagError::Invalid(format!(
                "trigger prefix automaton exceeds {MAX_DISPATCH_STATES} states"
            )));
        }
        Ok(Self {
            lowerer,
            tags,
            triggers: triggers.to_vec(),
            excludes: excludes.to_vec(),
            stop,
            max_chars,
            states: BTreeMap::new(),
        })
    }
    fn patterns(&self) -> impl Iterator<Item = &String> {
        self.triggers.iter().chain(&self.excludes)
    }
    fn start(&mut self, at_least_one: bool) -> Result<String, StructuralTagError> {
        if at_least_one {
            self.initial_tag_alternatives()
        } else if self.excludes.iter().any(String::is_empty) {
            Ok("\"\"".to_owned())
        } else {
            self.state("", self.max_chars)
        }
    }
    fn state(
        &mut self,
        prefix: &str,
        remaining: Option<u64>,
    ) -> Result<String, StructuralTagError> {
        let key = (prefix.to_owned(), remaining);
        if let Some(name) = self.states.get(&key) {
            return Ok(name.clone());
        }
        let name = format!("dispatch-{}", self.lowerer.next);
        self.lowerer.next += 1;
        self.states.insert(key, name.clone());
        let body = self.state_body(prefix, remaining)?;
        self.lowerer.rules.push((name.clone(), body));
        Ok(name)
    }
    fn state_body(
        &mut self,
        prefix: &str,
        remaining: Option<u64>,
    ) -> Result<String, StructuralTagError> {
        if remaining == Some(0) {
            return Ok("\"\"".to_owned());
        }
        let next_remaining = remaining.map(|value| value - 1);
        // Transition over the FULL pattern alphabet, not just the chars that
        // extend the current prefix: an alphabet char that fails to extend
        // still needs its KMP fallback state (longest_prefix_suffix), because
        // it may itself begin a new partial match. ("bbypass" against
        // exclusion "bypass" must reject — the second 'b' restarts the
        // match; a reset-to-empty fallback would wrongly accept.) The
        // catch-all class therefore excludes the entire alphabet and covers
        // only pattern-foreign chars, whose fallback is always the empty
        // prefix. Each state's branches are mutually exclusive (one class or
        // literal per alphabet char, one negated class for the rest), so the
        // automaton stays deterministic and live-stack pressure stays minimal.
        let mut alphabet = BTreeSet::new();
        for pattern in self.patterns() {
            alphabet.extend(pattern.chars());
        }
        let mut branches = Vec::new();
        let mut transitions: BTreeMap<String, Vec<char>> = BTreeMap::new();
        for ch in alphabet.iter().copied() {
            let emitted = format!("{prefix}{ch}");
            let completed_triggers = self
                .triggers
                .iter()
                .filter(|trigger| emitted.ends_with(*trigger))
                .cloned()
                .collect::<Vec<_>>();
            let completed_exclusion = self
                .excludes
                .iter()
                .any(|exclude| emitted.ends_with(exclude));
            if completed_triggers.is_empty() && !completed_exclusion {
                let patterns = self.patterns().cloned().collect::<Vec<_>>();
                let next = longest_prefix_suffix(&emitted, &patterns);
                transitions.entry(next).or_default().push(ch);
            } else if !completed_exclusion {
                for trigger in completed_triggers {
                    branches.extend(self.tag_for_trigger(&trigger)?);
                }
            }
        }
        for (next, chars) in transitions {
            branches.push(format!(
                "{} {}",
                char_expression(&chars, false)?,
                self.state(&next, next_remaining)?
            ));
        }
        branches.push(format!(
            "{} {}",
            char_expression(&alphabet.into_iter().collect::<Vec<_>>(), true)?,
            self.state("", next_remaining)?
        ));
        branches.push("\"\"".into());
        Ok(format!("( {} )", branches.join(" | ")))
    }
    fn initial_tag_alternatives(&mut self) -> Result<String, StructuralTagError> {
        let mut alternatives = Vec::new();
        for tag in self.tags.clone() {
            for end in &tag.end {
                let tail = if self.stop {
                    "\"\"".to_owned()
                } else {
                    let patterns = self.patterns().cloned().collect::<Vec<_>>();
                    self.state(
                        &end.as_text()
                            .map_or_else(String::new, |end| longest_prefix_suffix(end, &patterns)),
                        self.max_chars,
                    )?
                };
                alternatives.push(format!(
                    "{} {} {} {}",
                    tag.begin.expression()?,
                    tag.content,
                    end.expression()?,
                    tail
                ));
            }
        }
        if alternatives.is_empty() {
            return Err(StructuralTagError::Invalid(
                "at_least_one has no tag matching its trigger".into(),
            ));
        }
        Ok(if alternatives.len() == 1 {
            alternatives.remove(0)
        } else {
            format!("( {} )", alternatives.join(" | "))
        })
    }
    fn tag_for_trigger(&mut self, trigger: &str) -> Result<Vec<String>, StructuralTagError> {
        let mut out = Vec::new();
        for tag in self.tags.clone() {
            let Some(begin) = tag.begin.as_text() else {
                continue;
            };
            if begin.starts_with(trigger) {
                let suffix = &begin[trigger.len() - trigger.chars().last().unwrap().len_utf8()..];
                for end in &tag.end {
                    let tail = if self.stop {
                        "\"\"".to_owned()
                    } else {
                        let patterns = self.patterns().cloned().collect::<Vec<_>>();
                        self.state(
                            &end.as_text().map_or_else(String::new, |end| {
                                longest_prefix_suffix(end, &patterns)
                            }),
                            self.max_chars,
                        )?
                    };
                    out.push(format!(
                        "{} {} {} {}",
                        literal(suffix)?,
                        tag.content,
                        end.expression()?,
                        tail
                    ));
                }
            }
        }
        Ok(out)
    }
}

fn validate_trigger_topology(
    tags: &[TagSpec],
    triggers: &[String],
) -> Result<(), StructuralTagError> {
    if tags.is_empty() || triggers.is_empty() || triggers.iter().any(String::is_empty) {
        return Err(StructuralTagError::Invalid(
            "triggered tags require non-empty tags and non-empty triggers".into(),
        ));
    }
    for tag in tags {
        let Some(begin) = tag.begin.as_text() else {
            return Err(StructuralTagError::Invalid(
                "string-triggered tags require string begin terminals".into(),
            ));
        };
        let matching = triggers
            .iter()
            .filter(|trigger| begin.starts_with(trigger.as_str()))
            .count();
        if matching != 1 {
            return Err(StructuralTagError::Invalid(format!(
                "tag begin '{}' must match exactly one trigger",
                begin
            )));
        }
    }
    Ok(())
}

fn validate_scan_patterns(
    triggers: &[String],
    excludes: &[String],
) -> Result<(), StructuralTagError> {
    let patterns = triggers
        .iter()
        .map(|pattern| (pattern, "trigger"))
        .chain(excludes.iter().map(|pattern| (pattern, "exclusion")))
        .filter(|(pattern, _)| !pattern.is_empty())
        .collect::<Vec<_>>();
    for (pattern, kind) in &patterns {
        if pattern.contains('\0') {
            return Err(StructuralTagError::Invalid(format!(
                "{kind} patterns must not contain NUL"
            )));
        }
    }
    for (index, (left, _)) in patterns.iter().enumerate() {
        for (right, _) in patterns.iter().skip(index + 1) {
            if left.starts_with(right.as_str()) || right.starts_with(left.as_str()) {
                return Err(StructuralTagError::Invalid(format!(
                    "trigger and exclusion patterns must be distinct and prefix-free: {left:?}, {right:?}"
                )));
            }
        }
    }
    Ok(())
}

fn next_prefix_char(prefix: &str, trigger: &str) -> Option<char> {
    trigger
        .strip_prefix(prefix)
        .and_then(|tail| tail.chars().next())
}
fn longest_prefix_suffix(text: &str, triggers: &[String]) -> String {
    triggers
        .iter()
        .flat_map(|trigger| (0..trigger.len()).filter_map(|n| trigger.get(..n)))
        .filter(|prefix| text.ends_with(prefix))
        .max_by_key(|prefix| prefix.len())
        .unwrap_or("")
        .to_owned()
}

fn split_rules(source: &str) -> Result<Vec<(String, String)>, StructuralTagError> {
    source
        .lines()
        .map(|line| {
            line.split_once(" ::= ")
                .map(|(name, body)| (name.to_owned(), body.to_owned()))
                .ok_or_else(|| {
                    StructuralTagError::Invalid("serializer emitted malformed rule".into())
                })
        })
        .collect()
}
fn literal(value: &str) -> Result<String, StructuralTagError> {
    if value.contains('\0') {
        Err(StructuralTagError::Invalid(
            "NUL is not valid in a structural string literal".into(),
        ))
    } else {
        Ok(json_schema::format_literal(value))
    }
}
fn terminal_expression(terminals: &[Terminal]) -> Result<String, StructuralTagError> {
    if terminals.is_empty() {
        return Err(StructuralTagError::Invalid(
            "tag.end must not be an empty array".into(),
        ));
    }
    if terminals.len() == 1 {
        terminals[0].expression()
    } else {
        Ok(format!(
            "( {} )",
            terminals
                .iter()
                .map(Terminal::expression)
                .collect::<Result<Vec<_>, _>>()?
                .join(" | ")
        ))
    }
}

fn token_expression(token: u32) -> Result<String, StructuralTagError> {
    Ok(format!("<[{token}]>"))
}

fn exclude_tokens_expression(tokens: &[u32]) -> Result<String, StructuralTagError> {
    match tokens {
        [] => Ok("<[*]>".to_owned()),
        [token] => Ok(format!("!<[{token}]>")),
        tokens => Ok(format!(
            "!<[{}]>",
            tokens
                .iter()
                .map(u32::to_string)
                .collect::<Vec<_>>()
                .join(",")
        )),
    }
}

fn canonicalize_tokens(tokens: &mut Vec<u32>) {
    tokens.sort_unstable();
    tokens.dedup();
}
fn char_expression(chars: &[char], negated: bool) -> Result<String, StructuralTagError> {
    let mut inner = String::new();
    for ch in chars {
        match ch {
            '\\' => inner.push_str("\\\\"),
            ']' => inner.push_str("\\]"),
            '-' => inner.push_str("\\-"),
            '^' if !negated => inner.push_str("\\^"),
            '\0' => {
                return Err(StructuralTagError::Invalid(
                    "NUL trigger unsupported".into(),
                ));
            }
            _ => inner.push(*ch),
        }
    }
    Ok(if negated {
        format!("[^\\x00{}]", inner)
    } else if chars.len() == 1 {
        literal(&chars[0].to_string())?
    } else {
        format!("[{inner}]")
    })
}
fn object<'a>(
    value: &'a Value,
    context: &str,
) -> Result<&'a Map<String, Value>, StructuralTagError> {
    value
        .as_object()
        .ok_or_else(|| StructuralTagError::Invalid(format!("{context} must be an object")))
}
fn array<'a>(value: &'a Value, context: &str) -> Result<&'a [Value], StructuralTagError> {
    value
        .as_array()
        .map(Vec::as_slice)
        .ok_or_else(|| StructuralTagError::Invalid(format!("{context} must be an array")))
}
fn required<'a>(
    map: &'a Map<String, Value>,
    key: &str,
    context: &str,
) -> Result<&'a Value, StructuralTagError> {
    map.get(key)
        .ok_or_else(|| StructuralTagError::Invalid(format!("{context}.{key} is required")))
}
fn string<'a>(value: &'a Value, context: &str) -> Result<&'a str, StructuralTagError> {
    value
        .as_str()
        .ok_or_else(|| StructuralTagError::Invalid(format!("{context} must be a string")))
}
fn strings(value: &Value, context: &str) -> Result<Vec<String>, StructuralTagError> {
    array(value, context)?
        .iter()
        .map(|item| string(item, context).map(str::to_owned))
        .collect()
}
fn integer(value: &Value, context: &str) -> Result<i64, StructuralTagError> {
    value
        .as_i64()
        .ok_or_else(|| StructuralTagError::Invalid(format!("{context} must be an integer")))
}
fn bounded(value: &Value, context: &str) -> Result<u64, StructuralTagError> {
    let value = integer(value, context)?;
    if value < 0 || value as u64 > MAX_CHAR_BOUND {
        Err(StructuralTagError::Invalid(format!(
            "{context} must be in 0..={MAX_CHAR_BOUND}"
        )))
    } else {
        Ok(value as u64)
    }
}
fn expect_type(
    map: &Map<String, Value>,
    expected: &str,
    context: &str,
) -> Result<(), StructuralTagError> {
    if string(required(map, "type", context)?, &format!("{context}.type"))? == expected {
        Ok(())
    } else {
        Err(StructuralTagError::Invalid(format!(
            "{context}.type must be '{expected}'"
        )))
    }
}
fn optional_bool(
    map: &Map<String, Value>,
    key: &str,
    default: bool,
    context: &str,
) -> Result<bool, StructuralTagError> {
    match map.get(key) {
        None => Ok(default),
        Some(Value::Bool(value)) => Ok(*value),
        Some(_) => Err(StructuralTagError::Invalid(format!(
            "{context}.{key} must be a boolean"
        ))),
    }
}
fn exact_keys(
    map: &Map<String, Value>,
    allowed: &[&str],
    context: &str,
) -> Result<(), StructuralTagError> {
    if let Some(key) = map.keys().find(|key| !allowed.contains(&key.as_str())) {
        Err(StructuralTagError::Invalid(format!(
            "{context} contains unknown field '{key}'"
        )))
    } else {
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StructuralTagError {
    Invalid(String),
    NeedsTokenVocabulary(String),
    Unsupported(String),
}
impl std::fmt::Display for StructuralTagError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Invalid(message) | Self::Unsupported(message) => f.write_str(message),
            Self::NeedsTokenVocabulary(feature) => write!(
                f,
                "{feature} requires token/vocabulary-aware structural-tag compilation"
            ),
        }
    }
}
impl std::error::Error for StructuralTagError {}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn accepts(grammar: Grammar, output: &str) -> bool {
        let root = grammar.rule_id("root").unwrap();
        let mut runtime = crate::serve::api::grammar::GrammarRuntime::new(grammar, root).unwrap();
        runtime.accept_bytes(output.as_bytes()) && runtime.is_accepted()
    }

    #[test]
    fn current_composition_compiles_and_hybrid_fails_closed() {
        let value = json!({"type":"structural_tag","format":{"type":"sequence","elements":[{"type":"const_string","value":"<a>"},{"type":"repeat","min":1,"max":3,"content":{"type":"regex","pattern":"[0-9]"}},{"type":"const_string","value":"</a>"}]}});
        assert!(compile(&value).is_ok());
        let hybrid = json!({"type":"structural_tag","format":{"type":"const_string","value":"x"},"structures":[],"triggers":[]});
        assert!(matches!(
            lower_to_gbnf(&hybrid),
            Err(StructuralTagError::Invalid(_))
        ));
    }

    #[test]
    fn legacy_triggered_tags_enforce_schema_after_the_trigger() {
        let value = json!({"type":"structural_tag","structures":[{"begin":"<call>","schema":{"type":"object","properties":{"x":{"type":"string"}},"required":["x"],"additionalProperties":false},"end":"</call>"}],"triggers":["<call>"]});
        let source = lower_to_gbnf(&value).unwrap();
        assert!(source.contains("dispatch-"));
        assert!(accepts(
            parser::parse(&source).unwrap(),
            "preamble<call>{\"x\":\"ok\"}</call>tail"
        ));
        assert!(!accepts(
            parser::parse(&source).unwrap(),
            "preamble<call>not-json</call>tail"
        ));
    }

    #[test]
    fn forced_triggered_tag_and_separator_forms_compile() {
        let forced = json!({"type":"structural_tag","format":{"type":"triggered_tags","triggers":["<call>"],"tags":[{"type":"tag","begin":"<call>","content":{"type":"const_string","value":"ok"},"end":"</call>"}],"at_least_one":true,"stop_after_first":true}});
        assert!(compile(&forced).is_ok());
        let separated = json!({"type":"structural_tag","format":{"type":"tags_with_separator","tags":[{"type":"tag","begin":"<x>","content":{"type":"const_string","value":"x"},"end":"</x>"}],"separator":","}});
        assert!(compile(&separated).is_ok());
    }

    #[test]
    fn any_text_exclusions_and_character_budget_are_exact() {
        let value = json!({
            "type":"structural_tag",
            "format":{"type":"any_text","excludes":["STOP"],"max_chars":5}
        });
        let grammar = compile(&value).unwrap();
        assert!(accepts(grammar.clone(), "hello"));
        assert!(!accepts(grammar.clone(), "sixsix"));
        assert!(!accepts(grammar, "STOP"));
    }

    #[test]
    fn exclusion_automaton_rejects_overlapping_restarts() {
        // KMP fallback correctness: "bbypass" contains "bypass" starting at
        // the second byte; a reset-to-empty fallback would wrongly accept.
        let value = json!({
            "type":"structural_tag",
            "format":{"type":"any_text","excludes":["bypass"]}
        });
        let grammar = compile(&value).unwrap();
        assert!(!accepts(grammar.clone(), "bbypass"));
        assert!(!accepts(grammar.clone(), "a bypass here"));
        assert!(!accepts(grammar.clone(), "byp bypass"));
        assert!(!accepts(grammar.clone(), "bypass"));
        assert!(accepts(grammar.clone(), "bypa ss"));
        assert!(accepts(grammar.clone(), "byp"));
        assert!(accepts(grammar.clone(), ""));
        assert!(!accepts(grammar.clone(), "take the bypass lane"));
    }

    #[test]
    fn exclusion_automaton_is_unbounded_without_max_chars() {
        let value = json!({
            "type":"structural_tag",
            "format":{"type":"any_text","excludes":["bypass"]}
        });
        let grammar = compile(&value).unwrap();
        let long_clean = "lo".repeat(5000);
        assert!(accepts(grammar.clone(), &long_clean));
        let long_hit = format!("{}bypass", "lo".repeat(5000));
        assert!(!accepts(grammar.clone(), &long_hit));
        // restart storm at depth: every 'b' restarts the pattern
        let storm = format!("{}bypass", "b".repeat(3000));
        assert!(!accepts(grammar, &storm));
    }

    #[test]
    fn exclusion_automaton_covers_a_multi_pattern_lexicon() {
        let value = json!({
            "type":"structural_tag",
            "format":{"type":"any_text","excludes":["I cannot","I can't","I'm sorry","against my"]}
        });
        let grammar = compile(&value).unwrap();
        assert!(!accepts(grammar.clone(), "I cannot help with that."));
        assert!(!accepts(grammar.clone(), "well, I can't do that"));
        assert!(!accepts(grammar.clone(), "x I'm sorry x"));
        assert!(!accepts(grammar.clone(), "this goes against my guidelines"));
        assert!(accepts(grammar.clone(), "I can help with that."));
        assert!(accepts(grammar.clone(), "icannot is fine lowercase"));
        assert!(accepts(grammar.clone(), "sure, here is the answer"));
    }

    #[test]
    fn exclusion_automaton_bounded_mode_still_exact_after_alphabet_fix() {
        let value = json!({
            "type":"structural_tag",
            "format":{"type":"any_text","excludes":["ab"],"max_chars":3}
        });
        let grammar = compile(&value).unwrap();
        assert!(accepts(grammar.clone(), "aa"));      // clean, within budget
        assert!(accepts(grammar.clone(), "bba"));     // clean, at budget
        assert!(accepts(grammar.clone(), ""));        // empty
        assert!(!accepts(grammar.clone(), "aab"));    // contains ab at index 1
        assert!(!accepts(grammar.clone(), "aba"));    // contains ab at index 0
        assert!(!accepts(grammar.clone(), "aaaa"));   // over budget
    }

    #[test]
    fn string_dispatch_constrains_matched_content_and_honors_exclusions() {
        let value = json!({
            "type":"structural_tag",
            "format":{
                "type":"dispatch",
                "rules":[["<call>",{"type":"const_string","value":"ok"}]],
                "loop":true,
                "excludes":["BLOCK"]
            }
        });
        let grammar = compile(&value).unwrap();
        assert!(accepts(grammar.clone(), "free<call>oktail"));
        assert!(!accepts(grammar.clone(), "free<call>nope"));
        assert!(!accepts(grammar, "freeBLOCKtail"));
    }

    #[test]
    fn triggered_tags_honor_free_text_exclusions() {
        let value = json!({
            "type":"structural_tag",
            "format":{
                "type":"triggered_tags",
                "triggers":["<call>"],
                "tags":[{"type":"tag","begin":"<call>","content":{"type":"const_string","value":"ok"},"end":"</call>"}],
                "excludes":["BLOCK"]
            }
        });
        let grammar = compile(&value).unwrap();
        assert!(accepts(grammar.clone(), "free<call>ok</call>tail"));
        assert!(!accepts(grammar, "freeBLOCKtail"));
    }

    #[test]
    fn string_triggered_tags_keep_xgrammar_alternative_string_endings() {
        let value = json!({
            "type":"structural_tag",
            "format":{
                "type":"triggered_tags",
                "triggers":["<call>"],
                "tags":[{"type":"tag","begin":"<call>","content":{"type":"const_string","value":"ok"},"end":["</call>","</alt>"]}]
            }
        });
        let grammar = compile(&value).unwrap();
        assert!(accepts(grammar.clone(), "before<call>ok</call>after"));
        assert!(accepts(grammar, "before<call>ok</alt>after"));
    }

    #[test]
    fn json_schema_defaults_to_xgrammar_declaration_order() {
        let schema = json!({
            "type": "object",
            "properties": {"second": {"type": "integer"}, "first": {"type": "string"}},
            "required": ["second", "first"],
            "additionalProperties": false
        });
        let value =
            json!({"type":"structural_tag","format":{"type":"json_schema","json_schema":schema}});
        let grammar = compile(&value).unwrap();
        assert!(accepts(grammar.clone(), r#"{"second":1,"first":"ok"}"#));
        assert!(!accepts(grammar, r#"{"first":"ok","second":1}"#));

        let unordered = json!({
            "type":"structural_tag",
            "format":{"type":"json_schema","json_schema": schema, "any_order": true}
        });
        let grammar = compile(&unordered).unwrap();
        assert!(accepts(grammar.clone(), r#"{"first":"ok","second":1}"#));
        assert!(
            accepts(grammar.clone(), r#"{"second":1,"second":2}"#),
            "XGrammar any_order deliberately permits duplicates and does not track required-key identity"
        );
        assert!(!accepts(grammar.clone(), r#"{"second":1}"#));
        assert!(!accepts(grammar, r#"{"unknown":1,"second":2}"#));
    }

    #[test]
    fn json_schema_any_order_relaxation_applies_to_nested_objects_and_entry_counts() {
        let value = json!({
            "type":"structural_tag",
            "format":{
                "type":"json_schema",
                "any_order":true,
                "json_schema":{
                    "type":"object",
                    "properties":{
                        "cfg":{
                            "type":"object",
                            "properties":{"a":{"type":"integer"},"b":{"type":"integer"}},
                            "required":["a","b"],
                            "minProperties":3,
                            "maxProperties":3,
                            "additionalProperties":false
                        }
                    },
                    "required":["cfg"],
                    "additionalProperties":false
                }
            }
        });
        let grammar = compile(&value).unwrap();
        assert!(accepts(grammar.clone(), r#"{"cfg":{"a":1,"a":2,"b":3}}"#));
        assert!(!accepts(grammar.clone(), r#"{"cfg":{"a":1,"b":2}}"#));
        assert!(!accepts(grammar, r#"{"cfg":{"a":1,"a":2,"b":3,"b":4}}"#));
    }

    #[test]
    fn json_schema_bounds_whitespace_and_rejects_zero() {
        let value = json!({
            "type":"structural_tag",
            "format":{
                "type":"json_schema",
                "json_schema":{"type":"object","properties":{"x":{"type":"integer"}},"required":["x"],"additionalProperties":false},
                "max_whitespace_cnt":2
            }
        });
        assert!(accepts(compile(&value).unwrap(), r#"{  "x":  1  }"#));

        let zero = json!({"type":"structural_tag","format":{"type":"json_schema","json_schema":{},"max_whitespace_cnt":0}});
        assert!(matches!(
            lower_to_gbnf(&zero),
            Err(StructuralTagError::Invalid(_))
        ));
    }

    #[test]
    fn numeric_token_formats_lower_to_canonical_token_terminals() {
        let token = json!({"type":"structural_tag","format":{"type":"token","token":7}});
        assert!(lower_to_gbnf(&token).unwrap().contains("<[7]>"));

        let excluded = json!({"type":"structural_tag","format":{"type":"exclude_token","exclude_tokens":[9,2,9]}});
        assert!(lower_to_gbnf(&excluded).unwrap().contains("!<[2,9]>"));

        let bounded = json!({"type":"structural_tag","format":{"type":"any_tokens","exclude_tokens":[5],"max_tokens":2}});
        let source = lower_to_gbnf(&bounded).unwrap();
        assert!(source.contains("!<[5]>"));
        let root = parser::parse(&source).unwrap().rule_id("root").unwrap();
        let mut runtime =
            crate::serve::api::grammar::GrammarRuntime::new(parser::parse(&source).unwrap(), root)
                .unwrap();
        assert!(runtime.accept_token(1, b"irrelevant"));
        assert!(runtime.accept_token(2, b"irrelevant"));
        assert!(runtime.is_terminally_accepted());
        assert!(!runtime.accept_token(3, b"third token is over the bound"));

        let any_text = json!({"type":"structural_tag","format":{"type":"any_text","max_tokens":0}});
        assert!(accepts(compile(&any_text).unwrap(), ""));
    }

    #[test]
    fn token_strings_require_or_use_an_authoritative_resolver() {
        let value = json!({"type":"structural_tag","format":{"type":"token","token":"<open>"}});
        assert!(matches!(
            lower_to_gbnf(&value),
            Err(StructuralTagError::NeedsTokenVocabulary(_))
        ));
        assert!(lower_to_gbnf_with_token_resolver(&value, |token| {
            assert_eq!(token, "<open>");
            Ok(42)
        })
        .unwrap()
        .contains("<[42]>"));
    }

    #[test]
    fn token_tag_end_excludes_it_from_unbounded_any_tokens() {
        let value = json!({
            "type":"structural_tag",
            "format":{
                "type":"tag",
                "begin":{"type":"token","token":1},
                "content":{"type":"any_tokens","exclude_tokens":[8]},
                "end":{"type":"token","token":2}
            }
        });
        let source = lower_to_gbnf(&value).unwrap();
        assert!(source.contains("<[1]>"));
        assert!(source.contains("!<[2,8]>"));
        assert!(source.contains("<[2]>"));

        let grammar = compile(&value).unwrap();
        let root = grammar.rule_id("root").unwrap();
        let mut runtime = crate::serve::api::grammar::GrammarRuntime::new(grammar, root).unwrap();
        assert!(runtime.accept_token(1, b"start"));
        assert!(runtime.accept_token(7, b"body"));
        assert!(runtime.accept_token(2, b"end"));
        assert!(runtime.is_accepted());
    }

    #[test]
    fn token_triggered_and_dispatch_lower_with_trigger_safe_free_tokens() {
        let triggered = json!({
            "type":"structural_tag",
            "format":{
                "type":"token_triggered_tags",
                "trigger_tokens":[3],
                "exclude_tokens":[9],
                "tags":[{"type":"tag","begin":{"type":"token","token":3},"content":{"type":"const_string","value":"ok"},"end":{"type":"token","token":4}}]
            }
        });
        let source = lower_to_gbnf(&triggered).unwrap();
        assert!(source.contains("!<[3,9]>"));
        assert!(source.contains("<[3]>"));
        assert!(source.contains("<[4]>"));
        assert!(compile(&triggered).is_ok());

        let dispatch = json!({
            "type":"structural_tag",
            "format":{"type":"token_dispatch","rules":[[5,{"type":"const_string","value":"x"}]],"exclude_tokens":[8],"loop":false}
        });
        let source = lower_to_gbnf(&dispatch).unwrap();
        assert!(source.contains("!<[5,8]>"));
        assert!(source.contains("<[5]>"));
        assert!(compile(&dispatch).is_ok());
    }
}
