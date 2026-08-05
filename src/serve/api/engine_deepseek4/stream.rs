use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use tokio::sync::mpsc;

use crate::serve::api::engine::SamplingParams;
use crate::serve::api::grammar::GrammarRuntime;
use crate::serve::api::registry::{
    self, ModelRegistration, ReasoningSplitter, SplitSlot, ToolCallEvent, ToolCallSplitter,
};
use crate::serve::api::sse::{DeltaKind, GenerationEvent, StreamStats};

use super::sampling::{decode_token_limit, grammar_runtime, sample, sampler_config};
use super::Deepseek4LoadedModel;

fn send_visible(
    events: &mpsc::Sender<GenerationEvent>,
    event: GenerationEvent,
    request_started: Instant,
    first_visible_at: &mut Option<Duration>,
) -> Result<()> {
    events
        .blocking_send(event)
        .map_err(|_| anyhow::anyhow!("DeepSeek-V4 SSE client disconnected"))?;
    first_visible_at.get_or_insert_with(|| request_started.elapsed());
    Ok(())
}

fn emit_tool_block(
    registration: &ModelRegistration,
    body: &mut String,
    events: &mpsc::Sender<GenerationEvent>,
    next_index: &mut usize,
    request_started: Instant,
    first_visible_at: &mut Option<Duration>,
) -> Result<bool> {
    let calls = registry::parse_tool_call_bodies(registration, body)
        .context("parse DeepSeek-V4 DSML tool call block")?;
    for call in calls {
        let index = *next_index;
        let id = format!("call_hf2q_{:016x}", next_call_id(index));
        send_visible(
            events,
            GenerationEvent::ToolCallDelta {
                index,
                id: Some(id),
                call_type: Some("function".into()),
                name: Some(call.name),
                arguments: Some(call.arguments_json),
            },
            request_started,
            first_visible_at,
        )?;
        *next_index += 1;
    }
    body.clear();
    Ok(*next_index > 0)
}

fn next_call_id(index: usize) -> u64 {
    static NEXT: AtomicU64 = AtomicU64::new(1);
    NEXT.fetch_add(1, Ordering::Relaxed) ^ (index as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15)
}

#[allow(clippy::too_many_arguments)]
fn route_tool_content(
    text: String,
    tools: &mut Option<ToolCallSplitter>,
    body: &mut String,
    index: &mut usize,
    saw: &mut bool,
    runtime: &mut Option<GrammarRuntime>,
    registration: Option<&ModelRegistration>,
    events: &mpsc::Sender<GenerationEvent>,
    request_started: Instant,
    first_visible_at: &mut Option<Duration>,
) -> Result<()> {
    if text.is_empty() {
        return Ok(());
    }
    let Some(splitter) = tools.as_mut() else {
        send_visible(
            events,
            GenerationEvent::Delta {
                kind: DeltaKind::Content,
                text,
            },
            request_started,
            first_visible_at,
        )?;
        return Ok(());
    };
    for event in splitter.feed(&text) {
        match event {
            ToolCallEvent::Content(text) => send_visible(
                events,
                GenerationEvent::Delta {
                    kind: DeltaKind::Content,
                    text,
                },
                request_started,
                first_visible_at,
            )?,
            ToolCallEvent::ToolCallOpen => {
                body.clear();
                if let Some(runtime) = runtime.as_mut() {
                    runtime.trigger();
                }
            }
            ToolCallEvent::ToolCallText(text) => body.push_str(&text),
            ToolCallEvent::ToolCallClose => {
                let registration =
                    registration.context("DeepSeek-V4 tool splitter lacks registration")?;
                *saw |= emit_tool_block(
                    registration,
                    body,
                    events,
                    index,
                    request_started,
                    first_visible_at,
                )?;
            }
        }
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn route_stream_fragment(
    fragment: &str,
    reasoning: &mut Option<ReasoningSplitter>,
    tools: &mut Option<ToolCallSplitter>,
    body: &mut String,
    index: &mut usize,
    saw: &mut bool,
    runtime: &mut Option<GrammarRuntime>,
    registration: Option<&ModelRegistration>,
    events: &mpsc::Sender<GenerationEvent>,
    request_started: Instant,
    first_visible_at: &mut Option<Duration>,
) -> Result<()> {
    if let Some(splitter) = reasoning.as_mut() {
        for (slot, text) in splitter.feed(fragment) {
            match slot {
                SplitSlot::Reasoning => send_visible(
                    events,
                    GenerationEvent::Delta {
                        kind: DeltaKind::Reasoning,
                        text,
                    },
                    request_started,
                    first_visible_at,
                )?,
                SplitSlot::Content => route_tool_content(
                    text,
                    tools,
                    body,
                    index,
                    saw,
                    runtime,
                    registration,
                    events,
                    request_started,
                    first_visible_at,
                )?,
            }
        }
    } else {
        route_tool_content(
            fragment.to_string(),
            tools,
            body,
            index,
            saw,
            runtime,
            registration,
            events,
            request_started,
            first_visible_at,
        )?;
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub fn generate_stream(
    loaded: &mut Deepseek4LoadedModel,
    prompt_tokens: &[u32],
    params: &SamplingParams,
    events: &mpsc::Sender<GenerationEvent>,
    registration: Option<&ModelRegistration>,
    cancellation_counter: Option<&AtomicU64>,
) {
    let run = (|| -> Result<()> {
        let request_started = Instant::now();
        let prefill_started = request_started;
        let (mut logits, cached_tokens) =
            loaded.prefill_suffix(prompt_tokens, || events.is_closed())?;
        let prefill_duration = prefill_started.elapsed();
        let sampler = sampler_config(params);
        let mut runtime = grammar_runtime(params)?;
        let mut reasoning = registration.and_then(|registration| {
            registry::make_reasoning_splitter(registration, params.reasoning_forced_open)
        });
        let mut tools = registration.and_then(ToolCallSplitter::from_registration);
        let mut tool_body = String::new();
        let mut tool_index = 0usize;
        let mut saw_tool = false;
        let mut first_visible_at = None;
        let max_tokens = decode_token_limit(
            params.max_tokens,
            prompt_tokens.len(),
            loaded.context_limit(),
        );
        let decode_started = Instant::now();
        let mut generated = Vec::with_capacity(max_tokens);
        let mut decoded_running = String::new();
        let mut finish_reason = "length";

        let tokenizer = loaded.tokenizer.clone();
        let mut decoder = tokenizer.decode_stream(false);
        for step in 0..max_tokens {
            if events.is_closed() {
                anyhow::bail!("DeepSeek-V4 SSE client disconnected");
            }
            let (token, _) = sample(loaded, &logits, params, &sampler, &generated, &mut runtime)?;
            if loaded.eos_token_ids.contains(&token) {
                finish_reason = "stop";
                break;
            }
            if runtime.as_ref().is_some_and(|runtime| runtime.is_dead()) {
                finish_reason = "stop";
                break;
            }
            generated.push(token);
            if let Some(fragment) = decoder
                .step(token)
                .map_err(|error| anyhow::anyhow!("decode DeepSeek-V4 token {token}: {error}"))?
            {
                decoded_running.push_str(&fragment);
                route_stream_fragment(
                    &fragment,
                    &mut reasoning,
                    &mut tools,
                    &mut tool_body,
                    &mut tool_index,
                    &mut saw_tool,
                    &mut runtime,
                    registration,
                    events,
                    request_started,
                    &mut first_visible_at,
                )?;
            }
            if params
                .stop_strings
                .iter()
                .any(|stop| !stop.is_empty() && decoded_running.contains(stop))
            {
                finish_reason = "stop";
                break;
            }
            if step + 1 < max_tokens {
                logits = loaded.commit_generated_token(token)?;
            }
        }

        if let Some(splitter) = reasoning.as_mut() {
            if let Some((slot, text)) = splitter.finish() {
                match slot {
                    SplitSlot::Reasoning => send_visible(
                        events,
                        GenerationEvent::Delta {
                            kind: DeltaKind::Reasoning,
                            text,
                        },
                        request_started,
                        &mut first_visible_at,
                    )?,
                    SplitSlot::Content => route_tool_content(
                        text,
                        &mut tools,
                        &mut tool_body,
                        &mut tool_index,
                        &mut saw_tool,
                        &mut runtime,
                        registration,
                        events,
                        request_started,
                        &mut first_visible_at,
                    )?,
                }
            }
        }
        if let Some(splitter) = tools.as_mut() {
            if let Some(event) = splitter.finish() {
                match event {
                    ToolCallEvent::Content(text) => send_visible(
                        events,
                        GenerationEvent::Delta {
                            kind: DeltaKind::Content,
                            text,
                        },
                        request_started,
                        &mut first_visible_at,
                    )?,
                    ToolCallEvent::ToolCallText(text) => {
                        tool_body.push_str(&text);
                        anyhow::bail!("DeepSeek-V4 generation ended inside a DSML tool block");
                    }
                    ToolCallEvent::ToolCallOpen | ToolCallEvent::ToolCallClose => {}
                }
            }
        }
        if saw_tool {
            finish_reason = "tool_calls";
        }
        let decode_duration = decode_started.elapsed();
        let semantic_ttft = first_visible_at.unwrap_or_else(|| request_started.elapsed());
        events
            .blocking_send(GenerationEvent::Done {
                finish_reason,
                prompt_tokens: prompt_tokens.len(),
                completion_tokens: generated.len(),
                stats: StreamStats {
                    prefill_time_secs: Some(prefill_duration.as_secs_f64()),
                    decode_time_secs: Some(decode_duration.as_secs_f64()),
                    total_time_secs: Some(
                        prefill_duration.as_secs_f64() + decode_duration.as_secs_f64(),
                    ),
                    time_to_first_token_ms: Some(semantic_ttft.as_secs_f64() * 1000.0),
                    prefill_tokens_per_sec: Some(
                        (prompt_tokens.len().saturating_sub(cached_tokens)) as f64
                            / prefill_duration.as_secs_f64().max(f64::EPSILON),
                    ),
                    decode_tokens_per_sec: Some(
                        generated.len() as f64 / decode_duration.as_secs_f64().max(f64::EPSILON),
                    ),
                    cached_prompt_tokens: Some(cached_tokens),
                    reasoning_tokens: None,
                    ..StreamStats::default()
                },
            })
            .map_err(|_| anyhow::anyhow!("DeepSeek-V4 SSE client disconnected"))?;
        Ok(())
    })();

    if let Err(error) = run {
        if error.to_string().contains("disconnected") || events.is_closed() {
            if let Some(counter) = cancellation_counter {
                counter.fetch_add(1, Ordering::Relaxed);
            }
            tracing::info!("DeepSeek-V4 SSE stream dropped; generation cancelled");
        } else {
            let _ = events.blocking_send(GenerationEvent::Error(format!("{error:#}")));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::send_visible;
    use crate::serve::api::sse::{DeltaKind, GenerationEvent};
    use std::time::{Duration, Instant};
    use tokio::sync::mpsc;

    #[test]
    fn semantic_ttft_is_recorded_once_on_the_first_visible_event() {
        let (events, mut receiver) = mpsc::channel(1);
        let request_started = Instant::now();
        let mut first_visible_at = None;

        send_visible(
            &events,
            GenerationEvent::Delta {
                kind: DeltaKind::Content,
                text: "first".into(),
            },
            request_started,
            &mut first_visible_at,
        )
        .expect("send first visible event");
        assert!(matches!(
            receiver.blocking_recv(),
            Some(GenerationEvent::Delta { text, .. }) if text == "first"
        ));
        let first = first_visible_at.expect("first visible timestamp");

        send_visible(
            &events,
            GenerationEvent::Delta {
                kind: DeltaKind::Content,
                text: "second".into(),
            },
            request_started
                .checked_sub(Duration::from_secs(1))
                .expect("earlier instant"),
            &mut first_visible_at,
        )
        .expect("send second visible event");
        assert_eq!(first_visible_at, Some(first));
    }
}
