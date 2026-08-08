use std::time::Instant;

use anyhow::{Context, Result};
use mlx_native::MlxBuffer;

use crate::serve::api::engine::{
    arm_lazy_tool_grammar, effective_repetition_penalty, GenerationResult, SamplingParams,
};
use crate::serve::api::grammar::GrammarRuntime;
use crate::serve::api::registry::{self, ModelRegistration, SplitSlot};
use crate::serve::sampler_pure;

use super::progress::RequestProgress;
use super::{release_completed_prefill_scratch, Deepseek4LoadedModel, RequestScratchGuard};

pub(super) fn sampler_config(params: &SamplingParams) -> sampler_pure::SamplingParams {
    sampler_pure::SamplingParams {
        temperature: params.temperature as f64,
        top_p: params.top_p as f64,
        top_k: params.top_k,
        min_p: params.min_p as f64,
        repetition_penalty: effective_repetition_penalty(params),
        max_tokens: params.max_tokens,
    }
}

pub(super) fn decode_token_limit(
    requested: usize,
    prompt_tokens: usize,
    context_limit: usize,
) -> usize {
    // The first sampled token consumes the prompt logits and is committed only
    // if another decode step follows. A cache with N free positions can thus
    // return N + 1 tokens without ever writing past its fixed capacity.
    requested.max(1).min(
        context_limit
            .saturating_sub(prompt_tokens)
            .saturating_add(1),
    )
}

pub(super) fn grammar_runtime(
    params: &SamplingParams,
    registration: Option<&ModelRegistration>,
) -> Result<Option<GrammarRuntime>> {
    let Some(grammar) = params.grammar.as_ref() else {
        return Ok(None);
    };
    let root = grammar
        .rule_id("root")
        .context("grammar has no root rule")?;
    let mut runtime = GrammarRuntime::new(grammar.clone(), root)
        .context("grammar runtime initialization failed")?;
    arm_lazy_tool_grammar(&mut runtime, params.grammar_kind, registration);
    Ok(Some(runtime))
}

fn greedy_grammar_token(
    values: &mut [f32],
    previous: &[u32],
    repetition_penalty: f64,
    runtime: &GrammarRuntime,
    token_bytes: &[Vec<u8>],
) -> u32 {
    crate::serve::api::grammar::mask::sample_greedy_valid_token(
        values,
        previous,
        repetition_penalty,
        runtime,
        token_bytes,
    )
}

pub(super) fn sample(
    loaded: &mut Deepseek4LoadedModel,
    logits: &MlxBuffer,
    params: &SamplingParams,
    sampler: &sampler_pure::SamplingParams,
    previous: &[u32],
    runtime: &mut Option<GrammarRuntime>,
) -> Result<(u32, Option<f32>)> {
    let needs_cpu = params.temperature > 0.0
        || params.top_k > 0
        || params.top_p < 1.0
        || params.repetition_penalty != 1.0
        || !params.logit_bias.is_empty()
        || runtime.is_some()
        || params.logprobs;
    if !needs_cpu {
        return loaded.model.greedy_token(logits).map(|token| (token, None));
    }
    let mut values = logits
        .as_slice::<f32>()
        .context("read DeepSeek-V4 logits")?
        .to_vec();
    for (&token, &bias) in &params.logit_bias {
        if let Some(logit) = values.get_mut(token as usize) {
            *logit += bias;
        }
    }
    let greedy_grammar = params.temperature < sampler_pure::SAMPLING_EPS as f32
        && !params.logprobs
        && runtime.is_some()
        && params.token_bytes.is_some();
    let (token, logprob) = if greedy_grammar {
        let token = greedy_grammar_token(
            &mut values,
            previous,
            sampler.repetition_penalty,
            runtime.as_ref().expect("checked above"),
            params.token_bytes.as_deref().expect("checked above"),
        );
        (token, None)
    } else if params.logprobs {
        if let (Some(runtime), Some(token_bytes)) =
            (runtime.as_ref(), params.token_bytes.as_deref())
        {
            crate::serve::api::grammar::mask::mask_invalid_tokens(
                runtime,
                token_bytes,
                &mut values,
            );
        }
        let (token, logprob) =
            sampler_pure::sample_token_with_logprob(&mut values, sampler, previous);
        (token, Some(logprob))
    } else {
        if let (Some(runtime), Some(token_bytes)) =
            (runtime.as_ref(), params.token_bytes.as_deref())
        {
            crate::serve::api::grammar::mask::mask_invalid_tokens(
                runtime,
                token_bytes,
                &mut values,
            );
        }
        (
            sampler_pure::sample_token(&mut values, sampler, previous),
            None,
        )
    };
    if let (Some(runtime), Some(token_bytes)) = (runtime.as_mut(), params.token_bytes.as_deref()) {
        if let Some(bytes) = token_bytes.get(token as usize) {
            if !bytes.is_empty() {
                runtime.accept_bytes(bytes);
            }
        }
    }
    Ok((token, logprob))
}

pub(super) fn split_reasoning(
    raw: &str,
    registration: Option<&ModelRegistration>,
    forced_open: bool,
) -> (String, Option<String>) {
    let Some(mut splitter) = registration
        .filter(|registration| registration.has_reasoning())
        .and_then(|registration| registry::make_reasoning_splitter(registration, forced_open))
    else {
        return (raw.to_string(), None);
    };
    let mut content = String::new();
    let mut reasoning = String::new();
    let mut append = |slot: SplitSlot, text: String| match slot {
        SplitSlot::Content => content.push_str(&text),
        SplitSlot::Reasoning => reasoning.push_str(&text),
    };
    for (slot, text) in splitter.feed(raw) {
        append(slot, text);
    }
    if let Some((slot, text)) = splitter.finish() {
        append(slot, text);
    }
    (content, (!reasoning.is_empty()).then_some(reasoning))
}

pub fn generate_once(
    loaded: &mut Deepseek4LoadedModel,
    prompt_tokens: &[u32],
    params: &SamplingParams,
    registration: Option<&ModelRegistration>,
) -> Result<GenerationResult> {
    let scratch_guard = RequestScratchGuard::new();
    let mut progress = RequestProgress::start("unary", prompt_tokens.len(), params.max_tokens);
    let prefill_started = Instant::now();
    let (mut logits, cached_tokens) =
        loaded.prefill_suffix(prompt_tokens, params.max_tokens, || false, &mut progress)?;
    let prefill_duration = prefill_started.elapsed();
    progress.finish_prefill(prefill_duration);
    release_completed_prefill_scratch();
    let sampler = sampler_config(params);
    let mut runtime = grammar_runtime(params, registration)?;
    let max_tokens = decode_token_limit(
        params.max_tokens,
        prompt_tokens.len(),
        loaded.context_limit(),
    );
    let decode_started = Instant::now();
    progress.start_decode();
    let mut generated = Vec::with_capacity(max_tokens);
    let mut logprobs = params.logprobs.then(|| Vec::with_capacity(max_tokens));
    let tokenizer = loaded.tokenizer.clone();
    let mut stream = tokenizer.decode_stream(false);
    let mut raw = String::new();
    let mut finish_reason = "length";

    for step in 0..max_tokens {
        let (token, logprob) = sample(loaded, &logits, params, &sampler, &generated, &mut runtime)?;
        if loaded.eos_token_ids.contains(&token) {
            finish_reason = "stop";
            break;
        }
        if runtime.as_ref().is_some_and(|runtime| runtime.is_dead()) {
            finish_reason = "stop";
            break;
        }
        generated.push(token);
        if let (Some(values), Some(value)) = (logprobs.as_mut(), logprob) {
            values.push(value);
        }
        if let Some(fragment) = stream
            .step(token)
            .map_err(|error| anyhow::anyhow!("decode DeepSeek-V4 token {token}: {error}"))?
        {
            raw.push_str(&fragment);
        }
        if params
            .stop_strings
            .iter()
            .any(|stop| !stop.is_empty() && raw.contains(stop))
        {
            finish_reason = "stop";
            break;
        }
        if step + 1 < max_tokens {
            logits = loaded.commit_generated_token(token)?;
        }
        progress.advance_decode(generated.len());
    }

    let (text, reasoning_text) = split_reasoning(&raw, registration, params.reasoning_forced_open);
    progress.complete(finish_reason, generated.len(), None);
    scratch_guard.complete();
    Ok(GenerationResult {
        text,
        reasoning_text,
        prompt_tokens: prompt_tokens.len(),
        completion_tokens: generated.len(),
        reasoning_tokens: None,
        finish_reason,
        prefill_duration,
        decode_duration: decode_started.elapsed(),
        cached_tokens,
        logprobs,
    })
}

#[cfg(test)]
mod tests {
    use super::{decode_token_limit, greedy_grammar_token};

    #[test]
    fn greedy_grammar_fast_path_selects_highest_valid_token() {
        let grammar =
            crate::serve::api::grammar::parse("root ::= \"b\"\n").expect("parse literal grammar");
        let root = grammar.rule_id("root").expect("root rule");
        let runtime = crate::serve::api::grammar::GrammarRuntime::new(grammar, root)
            .expect("grammar runtime");
        let mut logits = vec![10.0, 9.0, 8.0];
        let bytes = vec![b"a".to_vec(), b"b".to_vec(), b"c".to_vec()];
        assert_eq!(
            greedy_grammar_token(&mut logits, &[], 1.0, &runtime, &bytes),
            1
        );
    }

    #[test]
    fn decode_limit_never_commits_beyond_fixed_context() {
        assert_eq!(decode_token_limit(128, 120, 128), 9);
        assert_eq!(decode_token_limit(128, 128, 128), 1);
        assert_eq!(decode_token_limit(0, 64, 128), 1);
        assert_eq!(decode_token_limit(4, 64, 128), 4);
    }
}
