use serde::de::{MapAccess, SeqAccess, Visitor};
use serde::{Deserialize, Deserializer};
use std::fmt;

/// JSON value retaining object insertion order, matching Python's
/// `json.loads` + `json.dumps` behavior used by the official encoder.
#[derive(Debug, Clone, PartialEq)]
pub enum OrderedValue {
    Null,
    Bool(bool),
    Number(serde_json::Number),
    String(String),
    Array(Vec<OrderedValue>),
    Object(Vec<(String, OrderedValue)>),
}

impl<'de> Deserialize<'de> for OrderedValue {
    fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        struct V;
        impl<'de> Visitor<'de> for V {
            type Value = OrderedValue;
            fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                f.write_str("JSON")
            }
            fn visit_unit<E>(self) -> Result<Self::Value, E> {
                Ok(OrderedValue::Null)
            }
            fn visit_none<E>(self) -> Result<Self::Value, E> {
                Ok(OrderedValue::Null)
            }
            fn visit_bool<E>(self, v: bool) -> Result<Self::Value, E> {
                Ok(OrderedValue::Bool(v))
            }
            fn visit_i64<E>(self, v: i64) -> Result<Self::Value, E> {
                Ok(OrderedValue::Number(v.into()))
            }
            fn visit_u64<E>(self, v: u64) -> Result<Self::Value, E> {
                Ok(OrderedValue::Number(v.into()))
            }
            fn visit_f64<E: serde::de::Error>(self, v: f64) -> Result<Self::Value, E> {
                serde_json::Number::from_f64(v)
                    .map(OrderedValue::Number)
                    .ok_or_else(|| E::custom("non-finite number"))
            }
            fn visit_str<E>(self, v: &str) -> Result<Self::Value, E> {
                Ok(OrderedValue::String(v.into()))
            }
            fn visit_string<E>(self, v: String) -> Result<Self::Value, E> {
                Ok(OrderedValue::String(v))
            }
            fn visit_seq<A: SeqAccess<'de>>(self, mut a: A) -> Result<Self::Value, A::Error> {
                let mut out = Vec::new();
                while let Some(v) = a.next_element()? {
                    out.push(v);
                }
                Ok(OrderedValue::Array(out))
            }
            fn visit_map<A: MapAccess<'de>>(self, mut a: A) -> Result<Self::Value, A::Error> {
                let mut out = Vec::new();
                while let Some((k, v)) = a.next_entry()? {
                    out.push((k, v));
                }
                Ok(OrderedValue::Object(out))
            }
        }
        d.deserialize_any(V)
    }
}

impl OrderedValue {
    pub(super) fn python_json(&self) -> String {
        match self {
            Self::Null => "null".into(),
            Self::Bool(v) => v.to_string(),
            Self::Number(v) => v.to_string(),
            Self::String(v) => serde_json::to_string(v).expect("string JSON"),
            Self::Array(v) => format!(
                "[{}]",
                v.iter()
                    .map(Self::python_json)
                    .collect::<Vec<_>>()
                    .join(", ")
            ),
            Self::Object(v) => format!(
                "{{{}}}",
                v.iter()
                    .map(|(k, v)| format!(
                        "{}: {}",
                        serde_json::to_string(k).expect("key JSON"),
                        v.python_json()
                    ))
                    .collect::<Vec<_>>()
                    .join(", ")
            ),
        }
    }
    pub(super) fn text(&self) -> Option<&str> {
        if let Self::String(v) = self {
            Some(v)
        } else {
            None
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct ToolDefinition {
    /// Ordered JSON because the official encoder serializes the complete
    /// OpenAI function object, including extension fields.
    pub function: OrderedValue,
}

impl ToolDefinition {
    pub(super) fn schema_json(&self) -> String {
        self.function.python_json()
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct ToolCallFunction {
    pub name: String,
    pub arguments: String,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ToolCall {
    #[serde(default)]
    pub id: Option<String>,
    pub function: ToolCallFunction,
}

#[derive(Debug, Clone, Deserialize)]
pub struct Message {
    pub role: String,
    #[serde(default)]
    pub content: Option<OrderedValue>,
    #[serde(default)]
    pub tools: Vec<ToolDefinition>,
    #[serde(default)]
    pub response_format: Option<OrderedValue>,
    #[serde(default)]
    pub tool_calls: Vec<ToolCall>,
    #[serde(default)]
    pub reasoning_content: Option<String>,
    #[serde(default)]
    pub tool_call_id: Option<String>,
    #[serde(default)]
    pub task: Option<String>,
    #[serde(default)]
    pub wo_eos: bool,
    #[serde(skip)]
    pub(super) content_blocks: Vec<ContentBlock>,
}

#[derive(Debug, Clone)]
pub(super) enum ContentBlock {
    Text(String),
    ToolResult { id: String, content: String },
}
