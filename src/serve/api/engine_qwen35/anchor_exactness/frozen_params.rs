//! Lossless test-only admission snapshot, captured from actual HTTP preparation.
//! Compiled grammar is retained, not independently recompiled or guessed from defaults.
use super::*;
use std::{collections::BTreeMap, sync::Arc};
use crate::serve::api::{engine::{GrammarKind, ToolCallPolicy}, grammar::{Grammar, GretElement, GretType, LazyGrammarConfig}, registry, schema::Tool};

pub(crate) struct TokenTable { pub(crate) sha256: String, pub(crate) values: Arc<Vec<Vec<u8>>> }
impl TokenTable {
    pub(crate) fn actual(values: Arc<Vec<Vec<u8>>>) -> Result<Self> {
        Ok(Self { sha256: hash(&serde_json::to_vec(values.as_ref())?), values })
    }
    pub(crate) fn load(binding: &acquisition::FileBinding) -> Result<Self> {
        let raw = acquisition::checked_file(binding, 128 * 1024 * 1024)?;
        let table = Self { sha256: binding.sha256.clone(), values: Arc::new(serde_json::from_slice(&raw)?) };
        ensure!(hash(&serde_json::to_vec(table.values.as_ref())?) == binding.sha256,
            "token-table binding must use the exact captured compact JSON bytes");
        Ok(table)
    }
}
#[derive(Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct CompiledGrammar { rules: Vec<Vec<(u8, u32)>>, symbols: BTreeMap<String, u32> }
#[derive(Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct Lazy { token_triggers: Vec<u32>, trigger_patterns: Vec<String>, preserved_tokens: Vec<u32> }
#[derive(Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct FrozenParams {
    temperature: f32, top_p: f32, top_k: usize, repetition_penalty: f32,
    pub(crate) max_tokens: usize, stop_strings: Vec<String>, frequency_penalty: f32,
    presence_penalty: f32, seed: Option<u64>, min_p: f32, logit_bias: BTreeMap<u32, f32>,
    logprobs: bool, top_logprobs: u32, parallel_tool_calls: bool,
    grammar: Option<CompiledGrammar>, token_table_sha256: Option<String>,
    grammar_kind: String, grammar_lazy: Option<Lazy>, tool_call_policy: String,
    has_tool_argument_wire_kinds: bool, reasoning_forced_open: bool,
    thinking_token_budget: Option<usize>, reasoning_end_tokens: Option<Vec<u32>>,
    reasoning_close_tokens: Option<Vec<u32>>, pub(crate) stable_prompt_prefix_tokens: Option<usize>,
}
impl FrozenParams {
    pub(crate) fn actual(p: &SamplingParams, table: Option<&TokenTable>, tools: &[Tool]) -> Result<Self> {
        ensure!(p.gemma_full_input_checkpoint.is_none() && p.vision_fingerprint.is_none(),
            "original C2 W is Qwen text-only; non-text admission cannot be silently narrowed");
        let token_table_sha256 = match &p.token_bytes {
            Some(values) => { let table = table.context("actual grammar table missing")?;
                ensure!(Arc::ptr_eq(values, &table.values) || values.as_ref() == table.values.as_ref(), "grammar token table changed");
                Some(table.sha256.clone()) },
            None => None,
        };
        if let Some(actual) = &p.tool_argument_wire_kinds {
            let expected = registry::qwen35_tool_argument_wire_kinds(tools).map_err(anyhow::Error::msg)?;
            ensure!(actual.as_ref() == &expected, "actual wire map differs from exact request tools");
        }
        Ok(Self {
            temperature:p.temperature,top_p:p.top_p,top_k:p.top_k,repetition_penalty:p.repetition_penalty,
            max_tokens:p.max_tokens,stop_strings:p.stop_strings.clone(),frequency_penalty:p.frequency_penalty,
            presence_penalty:p.presence_penalty,seed:p.seed,min_p:p.min_p,
            logit_bias:p.logit_bias.iter().map(|(&k,&v)|(k,v)).collect(),logprobs:p.logprobs,
            top_logprobs:p.top_logprobs,parallel_tool_calls:p.parallel_tool_calls,
            grammar:p.grammar.as_ref().map(|g|CompiledGrammar { rules:g.rules.iter().map(|r|r.iter().map(|e|(e.ty as u8,e.value)).collect()).collect(), symbols:g.symbol_ids.iter().map(|(k,v)|(k.clone(),*v)).collect() }),
            token_table_sha256, grammar_kind:match p.grammar_kind { GrammarKind::ResponseFormat=>"ResponseFormat",GrammarKind::ToolCallBodyAuto=>"ToolCallBodyAuto",GrammarKind::ToolCallBodyRequired=>"ToolCallBodyRequired" }.into(),
            grammar_lazy:p.grammar_lazy.as_ref().map(|g|Lazy {token_triggers:g.token_triggers.clone(),trigger_patterns:g.trigger_patterns.clone(),preserved_tokens:g.preserved_tokens.clone()}),
            tool_call_policy:match p.tool_call_policy {ToolCallPolicy::Auto=>"Auto",ToolCallPolicy::AutoLazyGrammar=>"AutoLazyGrammar",ToolCallPolicy::Constrained=>"Constrained"}.into(),
            has_tool_argument_wire_kinds:p.tool_argument_wire_kinds.is_some(),reasoning_forced_open:p.reasoning_forced_open,
            thinking_token_budget:p.thinking_token_budget,reasoning_end_tokens:p.reasoning_end_tokens.as_deref().cloned(),
            reasoning_close_tokens:p.reasoning_close_tokens.as_deref().cloned(),stable_prompt_prefix_tokens:p.stable_prompt_prefix_tokens,
        })
    }
    pub(crate) fn restore(&self, table: Option<&TokenTable>, tools: &[Tool]) -> Result<SamplingParams> {
        let token_bytes = self.token_table_sha256.as_ref().map(|sha| {
            let table = table.context("bound token table missing")?;
            ensure!(sha == &table.sha256, "compiled grammar/table binding differs");
            Ok::<_,anyhow::Error>(table.values.clone())
        }).transpose()?;
        let grammar = self.grammar.as_ref().map(|g| Ok::<_,anyhow::Error>(Grammar {
            rules:g.rules.iter().map(|r|r.iter().map(|&(ty,value)|Ok(GretElement {ty:element(ty)?,value})).collect()).collect::<Result<Vec<_>>>()?,
            symbol_ids:g.symbols.iter().map(|(k,v)|(k.clone(),*v)).collect(),
        })).transpose()?;
        ensure!(grammar.is_none() || token_bytes.is_some(), "actual grammar lost its byte table");
        Ok(SamplingParams {
            temperature:self.temperature,top_p:self.top_p,top_k:self.top_k,repetition_penalty:self.repetition_penalty,
            max_tokens:self.max_tokens,stop_strings:self.stop_strings.clone(),frequency_penalty:self.frequency_penalty,
            presence_penalty:self.presence_penalty,seed:self.seed,min_p:self.min_p,
            logit_bias:self.logit_bias.iter().map(|(&k,&v)|(k,v)).collect(),logprobs:self.logprobs,
            top_logprobs:self.top_logprobs,parallel_tool_calls:self.parallel_tool_calls,grammar,token_bytes,
            grammar_kind:match self.grammar_kind.as_str() {"ResponseFormat"=>GrammarKind::ResponseFormat,"ToolCallBodyAuto"=>GrammarKind::ToolCallBodyAuto,"ToolCallBodyRequired"=>GrammarKind::ToolCallBodyRequired,_=>anyhow::bail!("unknown captured grammar kind")},
            grammar_lazy:self.grammar_lazy.as_ref().map(|g|LazyGrammarConfig {token_triggers:g.token_triggers.clone(),trigger_patterns:g.trigger_patterns.clone(),preserved_tokens:g.preserved_tokens.clone()}),
            tool_call_policy:match self.tool_call_policy.as_str() {"Auto"=>ToolCallPolicy::Auto,"AutoLazyGrammar"=>ToolCallPolicy::AutoLazyGrammar,"Constrained"=>ToolCallPolicy::Constrained,_=>anyhow::bail!("unknown captured tool policy")},
            tool_argument_wire_kinds:self.has_tool_argument_wire_kinds.then(||registry::qwen35_tool_argument_wire_kinds(tools).map(Arc::new).map_err(anyhow::Error::msg)).transpose()?,
            reasoning_forced_open:self.reasoning_forced_open,thinking_token_budget:self.thinking_token_budget,
            reasoning_end_tokens:self.reasoning_end_tokens.clone().map(Arc::new),reasoning_close_tokens:self.reasoning_close_tokens.clone().map(Arc::new),
            stable_prompt_prefix_tokens:self.stable_prompt_prefix_tokens,gemma_full_input_checkpoint:None,vision_fingerprint:None,
            #[cfg(test)]
            c2_ticket:None,
        })
    }
}
fn element(ty:u8)->Result<GretType> { Ok(match ty {
    0=>GretType::End,1=>GretType::Alt,2=>GretType::RuleRef,3=>GretType::Char,4=>GretType::CharNot,
    5=>GretType::CharRngUpper,6=>GretType::CharAlt,7=>GretType::CharAny,8=>GretType::Token,
    9=>GretType::TokenNot,10=>GretType::TokenAny,11=>GretType::TokenNotSet,12=>GretType::TokenSetMember,
    _=>anyhow::bail!("unknown actual compiled grammar element {ty}"),
}) }
