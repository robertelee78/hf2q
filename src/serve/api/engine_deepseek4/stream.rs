use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use tokio::sync::mpsc;

use crate::serve::api::engine::{
    is_fatal_command_buffer_error, SamplingParams, SerialStreamEnd, SerialStreamResult,
};
use crate::serve::api::engine_supervisor::EngineSupervisor;
use crate::serve::api::grammar::GrammarRuntime;
use crate::serve::api::registry::{
    self, ModelRegistration, ReasoningSplitter, SplitSlot, ToolCallEvent, ToolCallSplitter,
};
use crate::serve::api::sse::{DeltaKind, GenerationEvent, StreamStats};

use super::progress::RequestProgress;
use super::sampling::{decode_token_limit, grammar_runtime, sample, sampler_config};
use super::{release_completed_prefill_scratch, Deepseek4LoadedModel, RequestScratchGuard};

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

fn emit_or_buffer_content(
    text: String,
    pending_whitespace: &mut String,
    events: &mpsc::Sender<GenerationEvent>,
    request_started: Instant,
    first_visible_at: &mut Option<Duration>,
) -> Result<()> {
    if text.is_empty() {
        return Ok(());
    }
    if text.trim().is_empty() {
        pending_whitespace.push_str(&text);
        return Ok(());
    }

    if pending_whitespace.is_empty() {
        send_visible(
            events,
            GenerationEvent::Delta {
                kind: DeltaKind::Content,
                text,
            },
            request_started,
            first_visible_at,
        )
    } else {
        let mut visible = std::mem::take(pending_whitespace);
        visible.push_str(&text);
        send_visible(
            events,
            GenerationEvent::Delta {
                kind: DeltaKind::Content,
                text: visible,
            },
            request_started,
            first_visible_at,
        )
    }
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
        .with_context(|| format!("parse DeepSeek-V4 DSML tool call block: body={body:?}"))?;
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

fn trigger_tool_grammar_on_raw_marker(
    decoded: &str,
    runtime: &mut Option<GrammarRuntime>,
    registration: Option<&ModelRegistration>,
) -> Result<()> {
    let Some(runtime) = runtime
        .as_mut()
        .filter(|runtime| runtime.is_awaiting_trigger())
    else {
        return Ok(());
    };
    let Some(open) = registration.and_then(|registration| registration.tool_open) else {
        return Ok(());
    };
    let Some(position) = decoded.rfind(open) else {
        return Ok(());
    };

    runtime.trigger();
    let suffix = &decoded[position + open.len()..];
    anyhow::ensure!(
        suffix.is_empty() || runtime.accept_bytes(suffix.as_bytes()),
        "DeepSeek-V4 tool grammar rejected bytes decoded with its open marker"
    );
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn route_tool_content(
    text: String,
    tools: &mut Option<ToolCallSplitter>,
    body: &mut String,
    pending_whitespace: &mut String,
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
            ToolCallEvent::Content(text) => emit_or_buffer_content(
                text,
                pending_whitespace,
                events,
                request_started,
                first_visible_at,
            )?,
            ToolCallEvent::ToolCallOpen => {
                // DSML commonly begins with formatting newlines. They are not
                // semantic assistant content on a pure tool turn and OpenAI-
                // compatible clients expect content to be null/empty.
                pending_whitespace.clear();
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
    pending_whitespace: &mut String,
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
                    pending_whitespace,
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
            pending_whitespace,
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

/// Per-request DeepSeek stream protocol state, separated from the verifier
/// loop so SlotAware can interleave several generations without losing DSML
/// tool-call framing, reasoning boundaries, or semantic-TTFT accounting.
pub(super) struct StreamRouter {
    reasoning: Option<ReasoningSplitter>,
    tools: Option<ToolCallSplitter>,
    tool_body: String,
    pending_content_whitespace: String,
    tool_index: usize,
    saw_tool: bool,
    request_started: Instant,
    first_visible_at: Option<Duration>,
}

impl StreamRouter {
    pub(super) fn new(
        registration: Option<&ModelRegistration>,
        reasoning_forced_open: bool,
        request_started: Instant,
    ) -> Self {
        Self {
            reasoning: registration.and_then(|registration| {
                registry::make_reasoning_splitter(registration, reasoning_forced_open)
            }),
            tools: registration.and_then(ToolCallSplitter::from_registration),
            tool_body: String::new(),
            pending_content_whitespace: String::new(),
            tool_index: 0,
            saw_tool: false,
            request_started,
            first_visible_at: None,
        }
    }

    pub(super) fn push(
        &mut self,
        fragment: &str,
        decoded_running: &str,
        runtime: &mut Option<GrammarRuntime>,
        registration: Option<&ModelRegistration>,
        events: &mpsc::Sender<GenerationEvent>,
    ) -> Result<()> {
        trigger_tool_grammar_on_raw_marker(decoded_running, runtime, registration)?;
        route_stream_fragment(
            fragment,
            &mut self.reasoning,
            &mut self.tools,
            &mut self.tool_body,
            &mut self.pending_content_whitespace,
            &mut self.tool_index,
            &mut self.saw_tool,
            runtime,
            registration,
            events,
            self.request_started,
            &mut self.first_visible_at,
        )
    }

    pub(super) fn finish(
        &mut self,
        runtime: &mut Option<GrammarRuntime>,
        registration: Option<&ModelRegistration>,
        events: &mpsc::Sender<GenerationEvent>,
    ) -> Result<bool> {
        if let Some(splitter) = self.reasoning.as_mut() {
            if let Some((slot, text)) = splitter.finish() {
                match slot {
                    SplitSlot::Reasoning => send_visible(
                        events,
                        GenerationEvent::Delta {
                            kind: DeltaKind::Reasoning,
                            text,
                        },
                        self.request_started,
                        &mut self.first_visible_at,
                    )?,
                    SplitSlot::Content => route_tool_content(
                        text,
                        &mut self.tools,
                        &mut self.tool_body,
                        &mut self.pending_content_whitespace,
                        &mut self.tool_index,
                        &mut self.saw_tool,
                        runtime,
                        registration,
                        events,
                        self.request_started,
                        &mut self.first_visible_at,
                    )?,
                }
            }
        }
        if let Some(splitter) = self.tools.as_mut() {
            if let Some(event) = splitter.finish() {
                match event {
                    ToolCallEvent::Content(text) => emit_or_buffer_content(
                        text,
                        &mut self.pending_content_whitespace,
                        events,
                        self.request_started,
                        &mut self.first_visible_at,
                    )?,
                    ToolCallEvent::ToolCallText(text) => {
                        self.tool_body.push_str(&text);
                        anyhow::bail!("DeepSeek-V4 generation ended inside a DSML tool block");
                    }
                    ToolCallEvent::ToolCallOpen | ToolCallEvent::ToolCallClose => {}
                }
            }
        }
        if !self.saw_tool && !self.pending_content_whitespace.is_empty() {
            send_visible(
                events,
                GenerationEvent::Delta {
                    kind: DeltaKind::Content,
                    text: std::mem::take(&mut self.pending_content_whitespace),
                },
                self.request_started,
                &mut self.first_visible_at,
            )?;
        }
        Ok(self.saw_tool)
    }

    pub(super) fn first_visible_at(&self) -> Option<Duration> {
        self.first_visible_at
    }
}

#[allow(clippy::too_many_arguments)]
pub(in crate::serve::api) fn generate_stream(
    loaded: &mut Deepseek4LoadedModel,
    prompt_tokens: &[u32],
    params: &SamplingParams,
    events: &mpsc::Sender<GenerationEvent>,
    registration: Option<&ModelRegistration>,
    cancellation_counter: Option<&AtomicU64>,
    supervisor: &EngineSupervisor,
) -> SerialStreamResult {
    let scratch_guard = RequestScratchGuard::new();
    let mut progress = RequestProgress::start("stream", prompt_tokens.len(), params.max_tokens);
    let run = (|| -> Result<()> {
        let request_started = Instant::now();
        let prefill_started = request_started;
        let (mut logits, cached_tokens) = loaded.prefill_suffix(
            prompt_tokens,
            params.max_tokens,
            || events.is_closed(),
            &mut progress,
            supervisor,
        )?;
        let prefill_duration = prefill_started.elapsed();
        progress.finish_prefill(prefill_duration);
        release_completed_prefill_scratch();
        let sampler = sampler_config(params);
        let mut runtime = grammar_runtime(params, registration)?;
        let mut reasoning = registration.and_then(|registration| {
            registry::make_reasoning_splitter(registration, params.reasoning_forced_open)
        });
        let mut tools = registration.and_then(ToolCallSplitter::from_registration);
        let mut tool_body = String::new();
        let mut pending_content_whitespace = String::new();
        let mut tool_index = 0usize;
        let mut saw_tool = false;
        let mut first_visible_at = None;
        let max_tokens = decode_token_limit(
            params.max_tokens,
            prompt_tokens.len(),
            loaded.context_limit(),
        );
        let decode_started = Instant::now();
        progress.start_decode();
        let mut generated = Vec::with_capacity(max_tokens);
        let mut decoded_running = String::new();
        let mut finish_reason = "length";

        let tokenizer = loaded.tokenizer.clone();
        let mut decoder = tokenizer.decode_stream(false);
        for step in 0..max_tokens {
            if events.is_closed() {
                anyhow::bail!("DeepSeek-V4 SSE client disconnected");
            }
            let (token, _) = sample(
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
            if let Some(fragment) = decoder
                .step(token)
                .map_err(|error| anyhow::anyhow!("decode DeepSeek-V4 token {token}: {error}"))?
            {
                decoded_running.push_str(&fragment);
                // Activate the lazy tool grammar at the raw decoded marker,
                // before the reasoning splitter's boundary tail can delay the
                // structured ToolCallOpen event by several bytes.
                trigger_tool_grammar_on_raw_marker(&decoded_running, &mut runtime, registration)?;
                route_stream_fragment(
                    &fragment,
                    &mut reasoning,
                    &mut tools,
                    &mut tool_body,
                    &mut pending_content_whitespace,
                    &mut tool_index,
                    &mut saw_tool,
                    &mut runtime,
                    registration,
                    events,
                    request_started,
                    &mut first_visible_at,
                )?;
                if let Some(ttft) = first_visible_at {
                    progress.first_semantic_token(ttft);
                }
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
                logits = loaded.commit_generated_token(token, supervisor)?;
            }
            progress.advance_decode(generated.len());
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
                        &mut pending_content_whitespace,
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
                    ToolCallEvent::Content(text) => emit_or_buffer_content(
                        text,
                        &mut pending_content_whitespace,
                        events,
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
        if !saw_tool && !pending_content_whitespace.is_empty() {
            send_visible(
                events,
                GenerationEvent::Delta {
                    kind: DeltaKind::Content,
                    text: std::mem::take(&mut pending_content_whitespace),
                },
                request_started,
                &mut first_visible_at,
            )?;
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
        progress.complete(finish_reason, generated.len(), Some(semantic_ttft));
        Ok(())
    })();

    match run {
        Ok(()) => {
            scratch_guard.complete();
            Ok(SerialStreamEnd::TerminalSent)
        }
        Err(error)
            if events.is_closed()
                && supervisor.is_healthy()
                && !is_fatal_command_buffer_error(&error) =>
        {
            if let Some(counter) = cancellation_counter {
                counter.fetch_add(1, Ordering::Relaxed);
            }
            progress.cancelled();
            tracing::info!("DeepSeek-V4 SSE stream dropped; generation cancelled");
            Ok(SerialStreamEnd::ClientClosed)
        }
        Err(error) => {
            progress.failed(&error);
            Err(error)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{route_tool_content, send_visible, trigger_tool_grammar_on_raw_marker};
    use crate::serve::api::grammar::{self, GrammarRuntime};
    use crate::serve::api::registry::{self, ToolCallSplitter};
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

    #[test]
    fn whitespace_before_deepseek_tool_call_is_not_visible_content() {
        let registration =
            registry::find_for("DeepSeek-V4-Flash-0731").expect("DeepSeek-V4 registration");
        let mut tools = ToolCallSplitter::from_registration(&registration);
        let mut body = String::new();
        let mut pending_whitespace = String::new();
        let mut index = 0;
        let mut saw = false;
        let mut runtime = None;
        let (events, mut receiver) = mpsc::channel(4);
        let mut first_visible_at = None;
        let raw = "\n\n<｜DSML｜tool_calls>\n<｜DSML｜invoke name=\"read_file\">\n<｜DSML｜parameter name=\"path\" string=\"true\">/tmp/Cargo.toml</｜DSML｜parameter>\n</｜DSML｜invoke>\n</｜DSML｜tool_calls>";

        route_tool_content(
            raw.into(),
            &mut tools,
            &mut body,
            &mut pending_whitespace,
            &mut index,
            &mut saw,
            &mut runtime,
            Some(&registration),
            &events,
            Instant::now(),
            &mut first_visible_at,
        )
        .expect("route DSML tool call");

        assert!(pending_whitespace.is_empty());
        assert!(saw);
        assert!(matches!(
            receiver.try_recv(),
            Ok(GenerationEvent::ToolCallDelta { name: Some(name), .. }) if name == "read_file"
        ));
        assert!(
            receiver.try_recv().is_err(),
            "no content delta may precede the tool call"
        );
    }

    #[test]
    fn raw_tool_marker_triggers_lazy_grammar_before_splitter_tail_drains() {
        let registration =
            registry::find_for("DeepSeek-V4-Flash-0731").expect("DeepSeek-V4 registration");
        let grammar = grammar::parse("root ::= \"\\n\" \"<｜DSML｜invoke name=\\\"bash\\\">\"\n")
            .expect("parse synthetic DSML body grammar");
        let root = grammar.rule_id("root").expect("root rule");
        let mut grammar_runtime = GrammarRuntime::new(grammar, root).expect("grammar runtime");
        grammar_runtime.set_awaiting_trigger(true);
        let mut runtime = Some(grammar_runtime);

        trigger_tool_grammar_on_raw_marker(
            "preamble<｜DSML｜tool_calls>",
            &mut runtime,
            Some(&registration),
        )
        .expect("trigger raw tool marker");
        let runtime = runtime.as_mut().expect("runtime");
        assert!(!runtime.is_awaiting_trigger());
        assert!(runtime.accept_bytes(b"\n"));
        assert!(
            !runtime.accept_bytes("<｜DSML｜\n".as_bytes()),
            "the formerly unconstrained bare DSML prefix must be rejected"
        );
    }
}
