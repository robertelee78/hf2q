//! XGrammar structural-tag JSON lowered into the shared GBNF runtime.
//!
//! The accepted surface is intentionally closed: a malformed or token-aware
//! format is an error, never unconstrained output.  String-triggered tags are
//! compiled as a finite-state scanner so free text cannot consume a trigger.

use std::collections::{BTreeMap, BTreeSet};

use serde_json::{Map, Value};

use super::{json_schema, parser, regex_gbnf, serialize, Grammar};

const MAX_CHAR_BOUND: u64 = 2_000;
const MAX_DISPATCH_STATES: usize = 256;

/// Compile either the current XGrammar object or vLLM's legacy structure.
pub fn compile(payload: &Value) -> Result<Grammar, StructuralTagError> {
    let source = lower_to_gbnf(payload)?;
    parser::parse(&source).map_err(|error| {
        StructuralTagError::Invalid(format!("generated GBNF was invalid: {error}"))
    })
}

/// Lower either accepted structural-tag shape to standalone GBNF text.
pub fn lower_to_gbnf(payload: &Value) -> Result<String, StructuralTagError> {
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
    let mut lowerer = Lowerer::default();
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

#[derive(Default)]
struct Lowerer {
    next: usize,
    rules: Vec<(String, String)>,
}

impl Lowerer {
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
            "sequence" | "or" => self.list(map, kind),
            "optional" | "plus" | "star" => self.unary(map, kind),
            "repeat" => self.repeat(map),
            "tag" => self.tag(map),
            "triggered_tags" => self.triggered(map),
            "tags_with_separator" => self.tags_with_separator(map),
            "dispatch" => Err(StructuralTagError::Unsupported("string dispatch requires a lazy trigger runtime and is not representable by this GBNF-only foundation".into())),
            "token" | "exclude_token" | "any_tokens" | "token_triggered_tags" | "token_dispatch" => {
                Err(StructuralTagError::NeedsTokenVocabulary(kind.to_owned()))
            }
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
        if map
            .get("any_order")
            .is_some_and(|v| v == &Value::Bool(true))
        {
            return Err(StructuralTagError::Unsupported(
                "json_schema.any_order is not represented by the shared schema compiler".into(),
            ));
        }
        if map.contains_key("max_whitespace_cnt") {
            return Err(StructuralTagError::Unsupported(
                "json_schema.max_whitespace_cnt is not represented by the shared schema compiler"
                    .into(),
            ));
        }
        let schema = required(map, "json_schema", "json_schema")?;
        if !schema.is_object() && !schema.is_boolean() {
            return Err(StructuralTagError::Invalid(
                "json_schema.json_schema must be an object or boolean".into(),
            ));
        }
        let source = json_schema::schema_to_gbnf(schema).map_err(|error| {
            StructuralTagError::Invalid(format!("invalid structural-tag JSON Schema: {error}"))
        })?;
        self.embed(&source, "structural-tag JSON Schema")
    }

    fn any_text(&mut self, map: &Map<String, Value>) -> Result<String, StructuralTagError> {
        exact_keys(
            map,
            &["type", "excludes", "max_tokens", "max_chars"],
            "any_text",
        )?;
        if let Some(value) = map.get("max_tokens") {
            if !value.is_null() {
                return Err(StructuralTagError::NeedsTokenVocabulary(
                    "any_text.max_tokens".into(),
                ));
            }
        }
        if let Some(excludes) = map.get("excludes") {
            if !strings(excludes, "any_text.excludes")?.is_empty() {
                return Err(StructuralTagError::Unsupported(
                    "any_text.excludes requires negative string matching".into(),
                ));
            }
        }
        let body = match map.get("max_chars") {
            None | Some(Value::Null) => "[^\\x00]*".to_owned(),
            Some(value) => format!("[^\\x00]{{0,{}}}", bounded(value, "any_text.max_chars")?),
        };
        Ok(self.rule(body))
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
        let begin = string_or_token(required(map, "begin", "tag")?, "tag.begin")?;
        let content = self.format(required(map, "content", "tag")?)?;
        let end = endings(required(map, "end", "tag")?)?;
        Ok(self.rule(format!(
            "{} {content} {}",
            literal(&begin)?,
            end_expression(&end)?
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
                begin: string(
                    required(map, "begin", "legacy structure")?,
                    "legacy structure.begin",
                )?
                .to_owned(),
                content,
                end: string(
                    required(map, "end", "legacy structure")?,
                    "legacy structure.end",
                )?
                .to_owned(),
            });
        }
        self.string_triggered(tags, triggers, false, false)
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
        if let Some(excludes) = map.get("excludes") {
            if !strings(excludes, "triggered_tags.excludes")?.is_empty() {
                return Err(StructuralTagError::Unsupported(
                    "triggered_tags.excludes requires negative string matching".into(),
                ));
            }
        }
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
                begin: string_or_token(
                    required(tag, "begin", "triggered_tags tag")?,
                    "triggered_tags tag.begin",
                )?,
                content: self.format(required(tag, "content", "triggered_tags tag")?)?,
                end: single_end(required(tag, "end", "triggered_tags tag")?)?,
            });
        }
        self.string_triggered(
            tags,
            &triggers,
            optional_bool(map, "at_least_one", false, "triggered_tags")?,
            optional_bool(map, "stop_after_first", false, "triggered_tags")?,
        )
    }

    fn string_triggered(
        &mut self,
        tags: Vec<TagSpec>,
        triggers: &[String],
        at_least_one: bool,
        stop_after_first: bool,
    ) -> Result<String, StructuralTagError> {
        validate_trigger_topology(&tags, triggers)?;
        let mut scanner = Scanner::new(self, tags, triggers, stop_after_first)?;
        let free_start = scanner.state("")?;
        let required_start = if at_least_one {
            scanner.initial_tag_alternatives()?
        } else {
            free_start
        };
        Ok(self.rule(required_start))
    }
}

#[derive(Clone)]
struct TagSpec {
    begin: String,
    content: String,
    end: String,
}

struct Scanner<'a> {
    lowerer: &'a mut Lowerer,
    tags: Vec<TagSpec>,
    triggers: Vec<String>,
    stop: bool,
    states: BTreeMap<String, String>,
}

impl<'a> Scanner<'a> {
    fn new(
        lowerer: &'a mut Lowerer,
        tags: Vec<TagSpec>,
        triggers: &[String],
        stop: bool,
    ) -> Result<Self, StructuralTagError> {
        let state_count = triggers.iter().map(|t| t.chars().count()).sum::<usize>();
        if state_count > MAX_DISPATCH_STATES {
            return Err(StructuralTagError::Invalid(format!(
                "trigger prefix automaton exceeds {MAX_DISPATCH_STATES} states"
            )));
        }
        Ok(Self {
            lowerer,
            tags,
            triggers: triggers.to_vec(),
            stop,
            states: BTreeMap::new(),
        })
    }
    fn state(&mut self, prefix: &str) -> Result<String, StructuralTagError> {
        if let Some(name) = self.states.get(prefix) {
            return Ok(name.clone());
        }
        let name = format!("dispatch-{}", self.lowerer.next);
        self.lowerer.next += 1;
        self.states.insert(prefix.to_owned(), name.clone());
        let body = self.state_body(prefix)?;
        self.lowerer.rules.push((name.clone(), body));
        Ok(name)
    }
    fn state_body(&mut self, prefix: &str) -> Result<String, StructuralTagError> {
        let mut chars = BTreeSet::new();
        for trigger in &self.triggers {
            if let Some(ch) = next_prefix_char(prefix, trigger) {
                chars.insert(ch);
            }
        }
        let mut branches = Vec::new();
        let mut transitions: BTreeMap<String, Vec<char>> = BTreeMap::new();
        for ch in chars.iter().copied() {
            let emitted = format!("{prefix}{ch}");
            let completed = self
                .triggers
                .iter()
                .filter(|trigger| emitted.ends_with(*trigger))
                .cloned()
                .collect::<Vec<_>>();
            if completed.is_empty() {
                let next = longest_prefix_suffix(&emitted, &self.triggers);
                transitions.entry(next).or_default().push(ch);
            } else {
                for trigger in completed {
                    branches.extend(self.tag_for_trigger(&trigger)?);
                }
            }
        }
        for (next, chars) in transitions {
            branches.push(format!(
                "{} {}",
                char_expression(&chars, false)?,
                self.state(&next)?
            ));
        }
        branches.push(format!(
            "{} {}",
            char_expression(&chars.into_iter().collect::<Vec<_>>(), true)?,
            self.state("")?
        ));
        branches.push("\"\"".into());
        Ok(format!("( {} )", branches.join(" | ")))
    }
    fn initial_tag_alternatives(&mut self) -> Result<String, StructuralTagError> {
        let mut alternatives = Vec::new();
        for tag in self.tags.clone() {
            let tail = if self.stop {
                "\"\"".to_owned()
            } else {
                self.state(&longest_prefix_suffix(&tag.end, &self.triggers))?
            };
            alternatives.push(format!(
                "{} {} {} {}",
                literal(&tag.begin)?,
                tag.content,
                literal(&tag.end)?,
                tail
            ));
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
            if tag.begin.starts_with(trigger) {
                let suffix =
                    &tag.begin[trigger.len() - trigger.chars().last().unwrap().len_utf8()..];
                let tail = if self.stop {
                    "\"\"".to_owned()
                } else {
                    self.state(&longest_prefix_suffix(&tag.end, &self.triggers))?
                };
                out.push(format!(
                    "{} {} {} {}",
                    literal(suffix)?,
                    tag.content,
                    literal(&tag.end)?,
                    tail
                ));
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
        let matching = triggers
            .iter()
            .filter(|trigger| tag.begin.starts_with(trigger.as_str()))
            .count();
        if matching != 1 {
            return Err(StructuralTagError::Invalid(format!(
                "tag begin '{}' must match exactly one trigger",
                tag.begin
            )));
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
fn end_expression(ends: &[String]) -> Result<String, StructuralTagError> {
    if ends.len() == 1 {
        literal(&ends[0])
    } else {
        Ok(format!(
            "( {} )",
            ends.iter()
                .map(|end| literal(end))
                .collect::<Result<Vec<_>, _>>()?
                .join(" | ")
        ))
    }
}
fn single_end(value: &Value) -> Result<String, StructuralTagError> {
    let ends = endings(value)?;
    if ends.len() != 1 {
        Err(StructuralTagError::Unsupported(
            "triggered_tags requires a single string end".into(),
        ))
    } else {
        Ok(ends.into_iter().next().unwrap())
    }
}
fn endings(value: &Value) -> Result<Vec<String>, StructuralTagError> {
    match value {
        Value::String(s) => Ok(vec![s.clone()]),
        Value::Array(_) => strings(value, "tag.end"),
        Value::Object(_) => Err(StructuralTagError::NeedsTokenVocabulary(
            "tag.end token".into(),
        )),
        _ => Err(StructuralTagError::Invalid(
            "tag.end must be a string, string array, or token format".into(),
        )),
    }
}
fn string_or_token(value: &Value, context: &str) -> Result<String, StructuralTagError> {
    match value {
        Value::String(s) => Ok(s.clone()),
        Value::Object(_) => Err(StructuralTagError::NeedsTokenVocabulary(context.into())),
        _ => Err(StructuralTagError::Invalid(format!(
            "{context} must be a string or token format"
        ))),
    }
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
                ))
            }
            _ => inner.push(*ch),
        }
    }
    Ok(if negated {
        format!("[^{}]", inner)
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
    fn token_and_unbounded_token_budget_are_explicit_errors() {
        let token = json!({"type":"structural_tag","format":{"type":"token","token":1}});
        assert!(matches!(
            lower_to_gbnf(&token),
            Err(StructuralTagError::NeedsTokenVocabulary(_))
        ));
        let budget = json!({"type":"structural_tag","format":{"type":"any_text","max_tokens":1}});
        assert!(matches!(
            lower_to_gbnf(&budget),
            Err(StructuralTagError::NeedsTokenVocabulary(_))
        ));
    }
}
