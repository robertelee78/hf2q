use std::time::Instant;

use anyhow::{Context, Result};
use mlx_native::MlxBuffer;

use crate::serve::api::engine::{
    effective_repetition_penalty, grammar_runtime_for_request, sample_logits_with_grammar,
    GenerationResult, GrammarKind, SamplingParams,
};
use crate::serve::api::engine_supervisor::EngineSupervisor;
use crate::serve::api::grammar::GrammarRuntime;
use crate::serve::api::registry::{self, ModelRegistration, SplitSlot};
use crate::serve::sampler_pure;

use super::progress::RequestProgress;
use super::{
    release_completed_prefill_scratch, Deepseek4LoadedModel, RequestScratchGuard,
    GPU_TRANSACTION_TIMEOUT,
};

pub(super) fn sampler_config(params: &SamplingParams) -> sampler_pure::SamplingParams {
    sampler_pure::SamplingParams {
        temperature: params.temperature as f64,
        top_p: params.top_p as f64,
        top_k: params.top_k,
        min_p: params.min_p as f64,
        repetition_penalty: effective_repetition_penalty(params),
        max_tokens: params.max_tokens,
        seed: params.seed,
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
    grammar_runtime_for_request(params, registration)
}

/// A non-parallel DeepSeek tool grammar has no semantically useful next-token
/// forward once the closing token makes the runtime accepting. The final token
/// is already part of the response; like stop-string and max-token termination,
/// it need not be appended to KV unless another decode step will consume it.
pub(super) fn accepted_single_tool_call_is_terminal(
    params: &SamplingParams,
    runtime: Option<&GrammarRuntime>,
) -> bool {
    !params.parallel_tool_calls
        && matches!(
            params.grammar_kind,
            GrammarKind::ToolCallBodyAuto | GrammarKind::ToolCallBodyRequired
        )
        && params.tool_call_policy.enforces_body_grammar()
        && runtime.is_some_and(|runtime| {
            !runtime.is_awaiting_trigger() && !runtime.is_dead() && runtime.is_accepted()
        })
}

fn sample_cpu_logits(
    values: &mut [f32],
    params: &SamplingParams,
    sampler: &sampler_pure::SamplingParams,
    previous: &[u32],
    runtime: Option<&GrammarRuntime>,
) -> Result<(u32, Option<f32>)> {
    sample_logits_with_grammar(
        values,
        sampler,
        previous,
        runtime,
        params.token_bytes.as_deref().map(Vec::as_slice),
        params.logprobs,
    )
}

pub(super) fn sample(
    loaded: &mut Deepseek4LoadedModel,
    logits: &MlxBuffer,
    params: &SamplingParams,
    sampler: &sampler_pure::SamplingParams,
    previous: &[u32],
    runtime: &mut Option<GrammarRuntime>,
    supervisor: &EngineSupervisor,
) -> Result<(u32, Option<f32>)> {
    let needs_cpu = params.temperature > 0.0
        || params.top_k > 0
        || params.top_p < 1.0
        || params.repetition_penalty != 1.0
        || !params.logit_bias.is_empty()
        || runtime.is_some()
        || params.token_bytes.is_some()
        || params.logprobs;
    if !needs_cpu {
        return supervisor
            .run("deepseek4_greedy_sample", GPU_TRANSACTION_TIMEOUT, || {
                loaded.model.greedy_token(logits)
            })
            .map(|token| (token, None));
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
    let (token, logprob) =
        sample_cpu_logits(&mut values, params, sampler, previous, runtime.as_ref())?;
    if let (Some(runtime), Some(token_bytes)) = (runtime.as_mut(), params.token_bytes.as_deref()) {
        if let Some(bytes) = token_bytes.get(token as usize) {
            if !bytes.is_empty() {
                runtime.accept_bytes(bytes);
            }
        }
    }
    Ok((token, logprob))
}

/// Advance the suspended required-tool grammar with a server-forced
/// reasoning boundary token. Normal sampling performs this update inside
/// [`sample`]; a forced token must preserve the same runtime transition or
/// the DSML body would remain unconstrained after `</think>`.
pub(super) fn accept_forced_token(
    runtime: &mut Option<GrammarRuntime>,
    params: &SamplingParams,
    token: u32,
) -> Result<()> {
    let runtime = runtime
        .as_mut()
        .context("DeepSeek-V4 forced reasoning close requires a tool grammar runtime")?;
    let token_bytes = params
        .token_bytes
        .as_deref()
        .context("DeepSeek-V4 forced reasoning close requires the token byte table")?;
    let bytes = token_bytes
        .get(token as usize)
        .with_context(|| format!("DeepSeek-V4 forced token {token} is outside the vocabulary"))?;
    anyhow::ensure!(
        !bytes.is_empty(),
        "DeepSeek-V4 forced reasoning token {token} has no wire bytes"
    );
    runtime.accept_bytes(bytes);
    anyhow::ensure!(
        !runtime.is_dead(),
        "DeepSeek-V4 forced reasoning close made the tool grammar dead"
    );
    Ok(())
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

pub(in crate::serve::api) fn generate_once(
    loaded: &mut Deepseek4LoadedModel,
    prompt_tokens: &[u32],
    params: &SamplingParams,
    registration: Option<&ModelRegistration>,
    supervisor: &EngineSupervisor,
) -> Result<GenerationResult> {
    let scratch_guard = RequestScratchGuard::new();
    let mut progress = RequestProgress::start("unary", prompt_tokens.len(), params.max_tokens);
    let prefill_started = Instant::now();
    let (mut logits, cached_tokens) = loaded.prefill_suffix(
        prompt_tokens,
        params.max_tokens,
        || false,
        &mut progress,
        supervisor,
    )?;
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
        let (token, logprob) = sample(
            loaded,
            &logits,
            params,
            &sampler,
            &generated,
            &mut runtime,
            supervisor,
        )?;
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
        if step + 1 < max_tokens && accepted_single_tool_call_is_terminal(params, runtime.as_ref())
        {
            finish_reason = "stop";
            break;
        }
        if step + 1 < max_tokens {
            logits = loaded.commit_generated_token(token, supervisor)?;
        }
        progress.advance_decode(generated.len());
    }

    let (text, reasoning_text) = split_reasoning(&raw, registration, params.reasoning_forced_open);
    loaded.commit_request_anchor();
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
    use std::sync::Arc;

    use super::{
        accept_forced_token, accepted_single_tool_call_is_terminal, decode_token_limit,
        grammar_runtime, sample_cpu_logits, sampler_config,
    };
    use crate::serve::api::engine::{GrammarKind, SamplingParams, ToolCallPolicy};

    #[test]
    fn forced_reasoning_close_arms_required_tool_grammar() {
        let grammar =
            crate::serve::api::grammar::parse("root ::= \"b\"\n").expect("parse literal grammar");
        let mut token_bytes = vec![Vec::new(); 2];
        token_bytes[0] = b"</think>".to_vec();
        token_bytes[1] = b"b".to_vec();
        let params = SamplingParams {
            grammar: Some(grammar),
            grammar_kind: GrammarKind::ToolCallBodyRequired,
            token_bytes: Some(Arc::new(token_bytes)),
            reasoning_forced_open: true,
            ..SamplingParams::default()
        };
        let registration = crate::serve::api::registry::find_for("deepseek-v4-flash")
            .expect("DeepSeek registration");
        let mut runtime =
            grammar_runtime(&params, Some(&registration)).expect("build grammar runtime");
        assert!(runtime.as_ref().unwrap().is_awaiting_trigger());
        accept_forced_token(&mut runtime, &params, 0).expect("force reasoning close");
        assert!(!runtime.as_ref().unwrap().is_awaiting_trigger());
        runtime.as_mut().unwrap().accept_bytes(b"b");
        assert!(runtime.as_ref().unwrap().is_accepted());
    }

    #[test]
    fn agentic_grammar_contract_deepseek_requires_exactly_one_authoritative_table() {
        let grammar =
            crate::serve::api::grammar::parse("root ::= \"b\"\n").expect("parse literal grammar");
        let missing = SamplingParams {
            grammar: Some(grammar.clone()),
            ..SamplingParams::default()
        };
        assert!(grammar_runtime(&missing, None)
            .expect_err("grammar without table must fail")
            .to_string()
            .contains("authoritative token byte table"));

        let stray = SamplingParams {
            token_bytes: Some(Arc::new(vec![b"a".to_vec(), b"b".to_vec()])),
            ..SamplingParams::default()
        };
        assert!(grammar_runtime(&stray, None)
            .expect_err("table without grammar must fail")
            .to_string()
            .contains("without a grammar"));

        let valid = SamplingParams {
            grammar: Some(grammar),
            token_bytes: Some(Arc::new(vec![b"a".to_vec(), b"b".to_vec()])),
            ..SamplingParams::default()
        };
        assert!(grammar_runtime(&valid, None)
            .expect("matching grammar/table presence")
            .is_some());
    }

    #[test]
    fn agentic_grammar_contract_deepseek_rejects_short_and_long_token_tables() {
        let grammar =
            crate::serve::api::grammar::parse("root ::= \"b\"\n").expect("parse literal grammar");
        let make_params = |table: Vec<Vec<u8>>| SamplingParams {
            grammar: Some(grammar.clone()),
            token_bytes: Some(Arc::new(table)),
            ..SamplingParams::default()
        };

        for table in [
            vec![b"a".to_vec(), b"b".to_vec()],
            vec![b"a".to_vec(), b"b".to_vec(), b"c".to_vec(), b"d".to_vec()],
        ] {
            let params = make_params(table);
            let sampler = sampler_config(&params);
            let runtime = grammar_runtime(&params, None).expect("runtime");
            let mut logits = vec![10.0, 9.0, 8.0];
            assert!(
                sample_cpu_logits(&mut logits, &params, &sampler, &[], runtime.as_ref(),)
                    .expect_err("table length mismatch must fail")
                    .to_string()
                    .contains("logits vocabulary")
            );
        }

        let params = make_params(vec![b"a".to_vec(), b"b".to_vec(), b"c".to_vec()]);
        let sampler = sampler_config(&params);
        let runtime = grammar_runtime(&params, None).expect("runtime");
        let mut logits = vec![10.0, 9.0, 8.0];
        let (token, logprob) =
            sample_cpu_logits(&mut logits, &params, &sampler, &[], runtime.as_ref())
                .expect("exact table");
        assert_eq!(token, 1);
        assert_eq!(logprob, None);
    }

    #[test]
    fn decode_limit_never_commits_beyond_fixed_context() {
        assert_eq!(decode_token_limit(128, 120, 128), 9);
        assert_eq!(decode_token_limit(128, 128, 128), 1);
        assert_eq!(decode_token_limit(0, 64, 128), 1);
        assert_eq!(decode_token_limit(4, 64, 128), 4);
    }

    #[test]
    fn accepted_single_tool_grammar_skips_only_the_terminal_forward() {
        let grammar = crate::serve::api::grammar::parse("root ::= \"b\"\n").expect("parse grammar");
        let mut runtime = crate::serve::api::grammar::GrammarRuntime::new(
            grammar.clone(),
            grammar.rule_id("root").expect("root rule"),
        )
        .expect("runtime");
        let mut params = SamplingParams {
            parallel_tool_calls: false,
            grammar_kind: GrammarKind::ToolCallBodyRequired,
            tool_call_policy: ToolCallPolicy::Constrained,
            ..SamplingParams::default()
        };

        assert!(!accepted_single_tool_call_is_terminal(&params, None));
        assert!(!accepted_single_tool_call_is_terminal(
            &params,
            Some(&runtime)
        ));
        runtime.accept_bytes(b"b");
        assert!(accepted_single_tool_call_is_terminal(
            &params,
            Some(&runtime)
        ));

        params.parallel_tool_calls = true;
        assert!(!accepted_single_tool_call_is_terminal(
            &params,
            Some(&runtime)
        ));
        params.parallel_tool_calls = false;
        params.grammar_kind = GrammarKind::ResponseFormat;
        assert!(!accepted_single_tool_call_is_terminal(
            &params,
            Some(&runtime)
        ));
        params.grammar_kind = GrammarKind::ToolCallBodyAuto;
        params.tool_call_policy = ToolCallPolicy::Auto;
        assert!(!accepted_single_tool_call_is_terminal(
            &params,
            Some(&runtime)
        ));
        params.tool_call_policy = ToolCallPolicy::AutoLazyGrammar;
        runtime.set_awaiting_trigger(true);
        assert!(!accepted_single_tool_call_is_terminal(
            &params,
            Some(&runtime)
        ));
        runtime.set_awaiting_trigger(false);
        assert!(accepted_single_tool_call_is_terminal(
            &params,
            Some(&runtime)
        ));
        runtime.accept_bytes(b"x");
        assert!(runtime.is_dead());
        assert!(!accepted_single_tool_call_is_terminal(
            &params,
            Some(&runtime)
        ));
    }
}
