//! Test-only nativeDS input replay; same HTTP normalization/render/tokenize seams.
use super::*;
use anyhow::{ensure,Result};
pub(crate) fn replay(wire:&[u8],tokenizer:&tokenizers::Tokenizer,template:&str)->Result<(Vec<u32>,String)> {
    ensure!(template==crate::core::chat_templates::DEEPSEEK_V4_FLASH_0731,"exact native DS template");
    let value:serde_json::Value=serde_json::from_slice(wire)?;
    ensure!(value["model"]=="DeepSeek V4 Flash 0731" &&value["max_completion_tokens"]==8192
        &&value["hf2q_overflow_policy"]=="reject" &&value["reasoning_effort"]=="max"
        &&value["temperature"]==0,"retained canonical DS request controls");
    let req:schema::ChatCompletionRequest=serde_json::from_value(value)?;
    ensure!(!req.messages.is_empty() &&req.messages.iter().all(|m|
        m.content.as_ref().is_none_or(|c|c.image_urls().is_empty())),"text-only DS request");
    ensure!(matches!(grammar::request::validate_tool_request(&req).map_err(|_|anyhow::anyhow!("actual DS request/render/tokenize refused"))?,
        schema::ToolChoiceValue::Auto),"actual DS auto-tools");
    let tools=req.tools.as_deref().ok_or_else(||anyhow::anyhow!("actual tool definitions"))?;
    ensure!(tools.len()==3,"actual canonical three tools");
    let registration=registry::find_for_architecture("deepseek4").ok_or_else(||anyhow::anyhow!("DS registry"))?;
    let controls=reasoning_controls_from_request(&req).map_err(|_|anyhow::anyhow!("actual DS request/render/tokenize refused"))?;
    let thinking=resolve_api_enable_thinking(controls.thinking_enabled_override(),
        crate::serve::template_supports_enable_thinking(template));
    let kwargs=resolve_api_template_kwargs(Some(&registration),req.chat_template_kwargs.as_ref(),
        controls.deepseek_reasoning_effort_tier());
    ensure!(kwargs.as_ref().and_then(|k|k.get("reasoning_effort"))==Some(&serde_json::json!("max")),"DS max effort");
    let rendered=render_chat_prompt_or_400_with_generation_prompt(template,&req.messages,Some(tools),
        thinking,kwargs.as_ref(),true).map_err(|_|anyhow::anyhow!("actual DS request/render/tokenize refused"))?;
    let ids=tokenize_http_prompt(tokenizer,&rendered).map_err(|_|anyhow::anyhow!("actual DS request/render/tokenize refused"))?;
    ensure!(ids.len()>=144 &&ids.len()<=262144-8192,"actual DS complete rendered prompt bounds");
    Ok((ids,rendered))
}
