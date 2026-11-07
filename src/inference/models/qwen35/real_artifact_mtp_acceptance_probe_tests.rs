//! Offline, teacher-forced MTP acceptance probe on a recorded agent run.
//!
//! The probe measures how many MTP drafts greedy speculation would accept on
//! a real served workload without serving it again. It replays the committed
//! (spec-off) token streams of every agent of one run:
//!
//! 1. **Reconstruction (CPU).** Every turn's prompt ids are re-rendered from
//!    the archived `request.json` through the serving chat template and
//!    tokenizer of the artifact, with the handler's thinking and template
//!    kwargs resolution. Each request's sha256 must equal the client receipt
//!    and each prompt's length must equal the served `usage.prompt_tokens`.
//!    A turn's decision stream is the tail of the rendered conversation that
//!    appends its assembled assistant message: the `usage.completion_tokens`
//!    counted ids, then the terminal stop token, which usage omits (a
//!    terminal decision is never pushed into `generated_tokens`). For every
//!    turn but the last, the server's own prefix cache proves the counted
//!    ids: the next turn reported `cached_tokens == prompt + completion`,
//!    and its rendered prompt holds exactly these ids there. The stop token
//!    itself is the template's (`<|im_end|>`); the archive cannot show which
//!    stop token the model sampled.
//! 2. **Teacher forcing (GPU, one model load).** Each agent's final sequence
//!    (last prompt plus last completion; every earlier turn is a prefix of
//!    it) is forwarded with the server's prefill call on prompt segments and
//!    in completion chunks on completion segments, keeping the
//!    post-output-norm `h_nextn` rows. MTP K/V is caught up exactly as the
//!    server does (`process_target_batch`, hidden shifted one row).
//! 3. **Drafting.** At every completion position `s` (pending committed token
//!    `t[s]`, verifier hidden `h[s-1]`, MTP cursor `s`), the MTP head is
//!    chained like `k3_draft`: `forward_draft_greedy_for_token` at positions
//!    `s, s+1, s+2`, each step fed the previous draft and the normalized MTP
//!    hidden. The chain stops at the first draft that differs from the
//!    committed token; the draft K/V is discarded back to `s`. The run length
//!    `L(s)` (0..=3) is how many leading drafts the committed stream accepts.
//! 4. **Rounds.** For n-max n, a turn's rounds start at its first completion
//!    position and advance by `min(L(s), n) + 1` emitted tokens, capped at the
//!    terminal token. Mean accept length is emitted tokens per round;
//!    per-draft acceptance is accepted drafts over `n` drafts per round.
//!
//! Numerics caveat: the verifier rows here come from the prefill route
//! (batched rows), not the served n4 joint decode row, and MTP catch-up runs
//! one row at a time in completion segments. Drafts are compared with the
//! served committed tokens, so the transcript is the served one; only the
//! hidden rows feeding the drafts differ by route rounding. The JSON reports
//! how often the teacher-forced target argmax (no repetition penalty) agrees
//! with the committed token as a route-fidelity indicator.
//!
//! Run (CPU dry run, then the GPU probe):
//!
//! ```text
//! HF2Q_MTP_PROBE_GGUF=/path/model.gguf HF2Q_MTP_PROBE_RUN=/path/run \
//!   cargo test --release --bin hf2q mtp_probe_transcripts_reconstruct \
//!   -- --ignored --nocapture
//! HF2Q_MTP_PROBE_GGUF=/path/model.gguf HF2Q_MTP_PROBE_RUN=/path/run \
//!   HF2Q_MTP_PROBE_OUT=/path/out.json \
//!   cargo test --release --bin hf2q mtp_acceptance_probe_real_run \
//!   -- --ignored --nocapture
//! ```
//!
//! `HF2Q_MTP_PROBE_RUN` names a run directory holding `client/agent-*/` (or
//! the `client` directory itself). `HF2Q_MTP_PROBE_COMPLETION_CHUNK` sets the
//! completion-segment forward width (default 128; 1 runs every completion
//! row as a one-row forward).

use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{anyhow, ensure, Context, Result};
use mlx_native::MlxBuffer;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};

use super::kv_cache::HybridKvCache;
use super::model::Qwen35Model;
use super::spec_decode::{last_hidden_row, nth_hidden_row, positions_for_range};
use crate::serve::api::schema::{ChatCompletionRequest, ChatMessage};
use crate::serve::multi_seq_kv::SlotId;

const GGUF_ENV: &str = "HF2Q_MTP_PROBE_GGUF";
const RUN_ENV: &str = "HF2Q_MTP_PROBE_RUN";
const OUT_ENV: &str = "HF2Q_MTP_PROBE_OUT";
const COMPLETION_CHUNK_ENV: &str = "HF2Q_MTP_PROBE_COMPLETION_CHUNK";
const SCHEMA: &str = "hf2q.mtp-acceptance-probe.v1";
/// Deepest draft chain probed; n-max 1..=MAX_DEPTH are simulated from it.
const MAX_DEPTH: usize = 3;
const PROMPT_CHUNK: usize = 2048;
const DEFAULT_COMPLETION_CHUNK: usize = 128;
/// Top-level request keys the reconstruction knows how to honour. Any other
/// key (reasoning controls, template kwargs, sampling) could change the
/// render or the decision, so it refuses instead of guessing.
const REQUEST_KEYS: &[&str] = &[
    "model",
    "messages",
    "tools",
    "tool_choice",
    "max_completion_tokens",
    "logprobs",
    "hf2q_overflow_policy",
    "stream",
    "stream_options",
];

fn sha256_hex(bytes: &[u8]) -> String {
    hex::encode(Sha256::digest(bytes))
}

fn token_ids_sha256(tokens: &[u32]) -> String {
    let mut hash = Sha256::new();
    for token in tokens {
        hash.update(token.to_le_bytes());
    }
    hex::encode(hash.finalize())
}

fn env_path(name: &str) -> PathBuf {
    std::env::var_os(name)
        .map(PathBuf::from)
        .unwrap_or_else(|| panic!("the MTP acceptance probe requires {name}"))
}

// ---------------------------------------------------------------------------
// Transcript reconstruction (CPU)
// ---------------------------------------------------------------------------

/// The serving prompt path of the artifact: embedded template and serving
/// tokenizer, with the handler's thinking and template-kwargs resolution for
/// a request that carries no reasoning controls or kwargs.
struct ServingRenderer {
    template: String,
    tokenizer: tokenizers::Tokenizer,
    enable_thinking: bool,
    template_kwargs: Option<serde_json::Map<String, Value>>,
    /// The family's stop tokens: the terminal decision of every turn.
    stop_ids: Vec<u32>,
}

impl ServingRenderer {
    fn from_gguf(path: &Path) -> Result<Self> {
        let gguf = mlx_native::gguf::GgufFile::open(path)
            .map_err(|error| anyhow!("open GGUF {}: {error}", path.display()))?;
        let architecture = gguf
            .metadata_string("general.architecture")
            .context("GGUF general.architecture")?
            .to_string();
        let (tokenizer, _) = crate::serve::api::engine_qwen35::build_qwen35_serving_tokenizer(&gguf)?;
        let template = gguf
            .metadata_string("tokenizer.chat_template")
            .context("the probe needs the artifact's embedded chat template")?
            .to_string();
        let registration = crate::serve::api::registry::find_for_architecture(&architecture);
        let enable_thinking = crate::serve::api::handlers::resolve_api_enable_thinking(
            None,
            crate::serve::template_supports_enable_thinking(&template),
        );
        let template_kwargs = crate::serve::api::handlers::resolve_api_template_kwargs(
            registration.as_ref(),
            None,
            None,
        );
        let stop_ids = crate::serve::api::engine_qwen35::QWEN35_EOS_STOP_NAMES
            .iter()
            .filter_map(|name| tokenizer.token_to_id(name))
            .collect::<Vec<_>>();
        ensure!(!stop_ids.is_empty(), "tokenizer has no Qwen stop token");
        Ok(Self {
            template,
            tokenizer,
            enable_thinking,
            template_kwargs,
            stop_ids,
        })
    }

    fn tokens(
        &self,
        messages: &[ChatMessage],
        tools: Option<&[crate::serve::api::schema::Tool]>,
        add_generation_prompt: bool,
    ) -> Result<Vec<u32>> {
        let rendered = crate::serve::api::engine::render_chat_prompt_with_tools_generation_prompt(
            &self.template,
            messages,
            tools,
            self.enable_thinking,
            self.template_kwargs.as_ref(),
            add_generation_prompt,
        )?;
        let encoding = self
            .tokenizer
            .encode(rendered.as_str(), false)
            .map_err(|error| anyhow!("tokenize rendered prompt: {error}"))?;
        Ok(encoding.get_ids().to_vec())
    }
}

/// One turn's span inside its agent's final sequence.
#[derive(Clone, Debug)]
struct TurnSpan {
    turn: usize,
    prompt_len: usize,
    /// Decision stream length: the counted completion tokens plus the
    /// terminal stop decision, which `usage.completion_tokens` omits (a
    /// terminal token is never pushed into `generated_tokens`; after a tool
    /// call the stop token is the grammar-terminal one).
    completion_len: usize,
    usage_completion_tokens: usize,
    request_sha256: String,
    /// `server_prefix_cache` when the next turn's cached span proves the
    /// completion ids; `render` for the agent's last turn.
    completion_proof: &'static str,
}

impl TurnSpan {
    /// Index of the terminal stop decision.
    fn end(&self) -> usize {
        self.prompt_len + self.completion_len - 1
    }
}

struct AgentTranscript {
    agent: String,
    sequence: Vec<u32>,
    turns: Vec<TurnSpan>,
    summary_sha256: String,
}

fn usage_field(turn: &Value, pointer: &str) -> Result<usize> {
    turn.pointer(pointer)
        .and_then(Value::as_u64)
        .map(|value| value as usize)
        .with_context(|| format!("summary turn field {pointer}"))
}

fn reconstruct_agent(renderer: &ServingRenderer, agent_dir: &Path) -> Result<AgentTranscript> {
    let agent = agent_dir
        .file_name()
        .and_then(|name| name.to_str())
        .context("agent directory name")?
        .to_string();
    let summary_bytes = std::fs::read(agent_dir.join("summary.json"))
        .with_context(|| format!("read {}/summary.json", agent_dir.display()))?;
    let summary: Value = serde_json::from_slice(&summary_bytes)?;
    let turn_records = summary["turns"].as_array().context("summary turns")?;
    ensure!(!turn_records.is_empty(), "{agent}: no turns");

    let mut prompts: Vec<Vec<u32>> = Vec::new();
    let mut completions: Vec<Vec<u32>> = Vec::new();
    let mut message_counts: Vec<usize> = Vec::new();
    let mut assistants: Vec<ChatMessage> = Vec::new();
    let mut turns: Vec<TurnSpan> = Vec::new();
    for (index, record) in turn_records.iter().enumerate() {
        ensure!(
            record["turn"].as_u64() == Some(index as u64),
            "{agent}: summary turn {index} out of order"
        );
        let turn_dir = agent_dir.join(format!("turn-{index}"));
        let request_bytes = std::fs::read(turn_dir.join("request.json"))
            .with_context(|| format!("read {}/request.json", turn_dir.display()))?;
        let request_sha256 = sha256_hex(&request_bytes);
        ensure!(
            record["request_sha256"].as_str() == Some(request_sha256.as_str()),
            "{agent} turn {index}: request.json sha256 {request_sha256} differs from the receipt"
        );
        let raw: Value = serde_json::from_slice(&request_bytes)?;
        let keys = raw.as_object().context("request is a JSON object")?;
        for key in keys.keys() {
            ensure!(
                REQUEST_KEYS.contains(&key.as_str()),
                "{agent} turn {index}: request key {key:?} is outside the reconstructed surface"
            );
        }
        ensure!(
            raw.get("tool_choice").is_none_or(|choice| *choice == "auto"),
            "{agent} turn {index}: only tool_choice auto is reconstructed"
        );
        let request: ChatCompletionRequest = serde_json::from_slice(&request_bytes)?;
        let tools = request.tools.as_deref();
        let prompt_tokens = usage_field(record, "/usage/prompt_tokens")?;
        let completion_tokens = usage_field(record, "/usage/completion_tokens")?;
        let cached_tokens = usage_field(record, "/usage/prompt_tokens_details/cached_tokens")?;
        ensure!(completion_tokens > 0, "{agent} turn {index}: empty completion");

        let prompt = renderer.tokens(&request.messages, tools, true)?;
        ensure!(
            prompt.len() == prompt_tokens,
            "{agent} turn {index}: rendered prompt has {} tokens, served usage.prompt_tokens {prompt_tokens}",
            prompt.len()
        );

        let assistant: ChatMessage = serde_json::from_slice(
            &std::fs::read(turn_dir.join("assembled-assistant.json"))
                .with_context(|| format!("read {}/assembled-assistant.json", turn_dir.display()))?,
        )?;
        let mut conversation = request.messages.clone();
        conversation.push(assistant.clone());
        let rendered = renderer.tokens(&conversation, tools, false)?;
        ensure!(
            rendered.len() > prompt.len() + completion_tokens
                && rendered[..prompt.len()] == prompt[..],
            "{agent} turn {index}: the rendered assistant turn does not extend the prompt"
        );
        let completion = rendered[prompt.len()..=prompt.len() + completion_tokens].to_vec();
        let stop = *completion.last().expect("non-empty completion");
        ensure!(
            renderer.stop_ids.contains(&stop),
            "{agent} turn {index}: counted completion is followed by {stop}, not a stop token"
        );

        if index > 0 {
            let previous = index - 1;
            let span = prompts[previous].len() + turns[previous].usage_completion_tokens;
            ensure!(
                cached_tokens == span,
                "{agent} turn {index}: served cached_tokens {cached_tokens} != previous prompt+completion {span}"
            );
            ensure!(
                prompt.len() > span
                    && prompt[..prompts[previous].len()] == prompts[previous][..]
                    && prompt[prompts[previous].len()..span]
                        == completions[previous][..turns[previous].usage_completion_tokens],
                "{agent} turn {index}: prompt does not extend turn {previous}'s prompt and completion"
            );
            ensure!(
                request.messages.get(message_counts[previous]) == Some(&assistants[previous]),
                "{agent} turn {index}: echoed assistant message differs from turn {previous}'s assembled reply"
            );
        } else {
            ensure!(cached_tokens == 0, "{agent} turn 0: unexpected cached prefix");
        }

        turns.push(TurnSpan {
            turn: index,
            prompt_len: prompt.len(),
            completion_len: completion.len(),
            usage_completion_tokens: completion_tokens,
            request_sha256,
            completion_proof: "render",
        });
        if index > 0 {
            turns[index - 1].completion_proof = "server_prefix_cache";
        }
        message_counts.push(request.messages.len());
        assistants.push(assistant);
        prompts.push(prompt);
        completions.push(completion);
    }

    let mut sequence = prompts.last().expect("turns").clone();
    sequence.extend_from_slice(completions.last().expect("turns"));
    for (span, completion) in turns.iter().zip(&completions) {
        ensure!(
            sequence[span.prompt_len..=span.end()] == completion[..],
            "{agent} turn {}: completion is not embedded in the final sequence",
            span.turn
        );
    }
    Ok(AgentTranscript {
        agent,
        sequence,
        turns,
        summary_sha256: sha256_hex(&summary_bytes),
    })
}

fn client_dir(run: &Path) -> PathBuf {
    let nested = run.join("client");
    if nested.is_dir() {
        nested
    } else {
        run.to_path_buf()
    }
}

fn reconstruct_run(renderer: &ServingRenderer, run: &Path) -> Result<Vec<AgentTranscript>> {
    let client = client_dir(run);
    let mut agent_dirs = std::fs::read_dir(&client)
        .with_context(|| format!("read {}", client.display()))?
        .filter_map(|entry| entry.ok().map(|entry| entry.path()))
        .filter(|path| {
            path.is_dir()
                && path
                    .file_name()
                    .and_then(|name| name.to_str())
                    .is_some_and(|name| name.starts_with("agent-"))
        })
        .collect::<Vec<_>>();
    agent_dirs.sort();
    ensure!(!agent_dirs.is_empty(), "no agent-* directory under {}", client.display());
    agent_dirs
        .iter()
        .map(|dir| reconstruct_agent(renderer, dir))
        .collect()
}

fn transcript_evidence(renderer: &ServingRenderer, run: &Path, agents: &[AgentTranscript]) -> Value {
    let receipt_identity = std::fs::read(run.join("receipt.json"))
        .ok()
        .and_then(|bytes| serde_json::from_slice::<Value>(&bytes).ok())
        .map(|receipt| receipt["identity"].clone())
        .unwrap_or(Value::Null);
    json!({
        "run": run.display().to_string(),
        "run_identity": receipt_identity,
        "template_sha256": sha256_hex(renderer.template.as_bytes()),
        "enable_thinking": renderer.enable_thinking,
        "template_kwargs": renderer.template_kwargs,
        "agents": agents.iter().map(|agent| json!({
            "agent": agent.agent,
            "summary_sha256": agent.summary_sha256,
            "sequence_tokens": agent.sequence.len(),
            "sequence_token_ids_sha256": token_ids_sha256(&agent.sequence),
            "turns": agent.turns.iter().map(|span| json!({
                "turn": span.turn,
                "request_sha256": span.request_sha256,
                "prompt_tokens": span.prompt_len,
                "decision_tokens": span.completion_len,
                "usage_completion_tokens": span.usage_completion_tokens,
                "completion_proof": span.completion_proof,
            })).collect::<Vec<_>>(),
        })).collect::<Vec<_>>(),
    })
}

// ---------------------------------------------------------------------------
// Round simulation (pure)
// ---------------------------------------------------------------------------

/// Speculative rounds of one n-max over any number of turns.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct RoundTally {
    rounds: u64,
    /// Tokens emitted by rounds (accepted drafts plus the verifier's own
    /// token, capped at the terminal token).
    emitted: u64,
    accepted: u64,
    drafted: u64,
    /// Rounds by accepted drafts, index 0..=MAX_DEPTH.
    histogram: [u64; MAX_DEPTH + 1],
}

impl RoundTally {
    /// `runs[i]` is `L(s)` at the turn's i-th completion position; its length
    /// is the completion length minus one (the first completion token comes
    /// from prefill, the last is the terminal token).
    fn add_turn(&mut self, runs: &[u8], n_max: usize) {
        let positions = runs.len();
        let mut index = 0;
        while index < positions {
            let accepted = (runs[index] as usize).min(n_max);
            let emitted = (accepted + 1).min(positions - index);
            self.rounds += 1;
            self.emitted += emitted as u64;
            self.accepted += accepted as u64;
            self.drafted += n_max as u64;
            self.histogram[accepted] += 1;
            index += emitted;
        }
    }

    fn to_json(self, n_max: usize) -> Value {
        let ratio = |num: u64, den: u64| (den > 0).then(|| num as f64 / den as f64);
        json!({
            "n_max": n_max,
            "rounds": self.rounds,
            "emitted_tokens": self.emitted,
            "accepted_drafts": self.accepted,
            "drafted": self.drafted,
            "mean_accept_length": ratio(self.emitted, self.rounds),
            "per_draft_acceptance": ratio(self.accepted, self.drafted),
            "accepted_per_round": ratio(self.accepted, self.rounds),
            "accepted_histogram": &self.histogram[..=n_max],
        })
    }
}

/// Conditional acceptance of draft depth d given depths 1..d-1 accepted,
/// over every completion position whose remaining tokens reach depth d.
fn depth_acceptance(turn_runs: &[Vec<u8>]) -> Vec<Value> {
    (1..=MAX_DEPTH)
        .map(|depth| {
            let mut eligible = 0u64;
            let mut hits = 0u64;
            for runs in turn_runs {
                let positions = runs.len();
                for (index, &run) in runs.iter().enumerate() {
                    if positions - index >= depth && run as usize >= depth - 1 {
                        eligible += 1;
                        hits += u64::from(run as usize >= depth);
                    }
                }
            }
            json!({
                "depth": depth,
                "eligible": eligible,
                "accepted": hits,
                "conditional_acceptance": (eligible > 0).then(|| hits as f64 / eligible as f64),
            })
        })
        .collect()
}

fn round_summary(turn_runs: &[Vec<u8>]) -> Value {
    let by_n = (1..=MAX_DEPTH)
        .map(|n_max| {
            let mut tally = RoundTally::default();
            for runs in turn_runs {
                tally.add_turn(runs, n_max);
            }
            tally.to_json(n_max)
        })
        .collect::<Vec<_>>();
    json!({
        "by_n_max": by_n,
        "depth_conditional": depth_acceptance(turn_runs),
    })
}

// ---------------------------------------------------------------------------
// Teacher-forced probe (GPU)
// ---------------------------------------------------------------------------

struct TurnProbe {
    /// `L(s)` for s in `prompt_len..end`.
    runs: Vec<u8>,
    /// Drafts produced at each position, up to the first rejection.
    drafts: Vec<Vec<u32>>,
    /// Rows whose target argmax was compared with the committed next token.
    argmax_rows: u64,
    argmax_agree: u64,
}

fn argmax(row: &[f32]) -> u32 {
    let mut best = 0;
    for (index, &value) in row.iter().enumerate() {
        if value > row[best] {
            best = index;
        }
    }
    best as u32
}

/// The model the server would load from `path`: weights, MTP block and the
/// routing policy `hf2q serve` resolves from the artifact path.
fn load_serving_model(path: &Path) -> Result<Qwen35Model> {
    use crate::serve::header::LoadProgress;
    let gguf = mlx_native::gguf::GgufFile::open(path)
        .map_err(|error| anyhow!("open GGUF {}: {error}", path.display()))?;
    let mut progress = LoadProgress::new(false, 1, 0);
    let mut model = Qwen35Model::load_from_gguf(&gguf, &mut progress)?;
    model.bind_ggml_routing_policy(super::routing_policy::resolve_qwen_ggml_routing_policy(path));
    Ok(model)
}

fn mtp_cursor(cache: &HybridKvCache, slot: SlotId) -> Result<usize> {
    cache
        .mtp_slot
        .as_ref()
        .and_then(|mtp| mtp.current_len.get(slot.0 as usize).copied())
        .map(|len| len as usize)
        .context("MTP cursor")
}

/// Catch the MTP K/V up over `tokens` at `start..`, as the server does after
/// a prefill chunk or a verify window: row 0 consumes `pending`, row i the
/// target hidden row i-1.
#[allow(clippy::too_many_arguments)]
fn mtp_catch_up(
    model: &Qwen35Model,
    cache: &mut HybridKvCache,
    slot: SlotId,
    tokens: &[u32],
    start: usize,
    pending: Option<&MlxBuffer>,
    target_nextn: &MlxBuffer,
) -> Result<()> {
    let mtp = model.mtp.as_ref().context("artifact has no MTP block")?;
    ensure!(mtp_cursor(cache, slot)? == start, "MTP cursor is not at {start}");
    let positions = positions_for_range(start as i32, tokens.len());
    let shared_embed_rows = model.embed_tokens_gpu(tokens)?;
    model.with_gpu_cache_mut(|device, registry| {
        mtp.process_target_batch(
            tokens,
            pending,
            target_nextn,
            &shared_embed_rows,
            cache,
            slot,
            &positions,
            device,
            registry,
            &model.cfg,
        )
    })
}

/// `k3_draft`'s chain at position `s`, stopped at the first draft the
/// committed stream rejects; the draft K/V is discarded back to `s`.
fn draft_chain(
    model: &Qwen35Model,
    cache: &mut HybridKvCache,
    slot: SlotId,
    sequence: &[u32],
    s: usize,
    depth_cap: usize,
    verifier_hidden: &MlxBuffer,
) -> Result<Vec<u32>> {
    let mtp = model.mtp.as_ref().context("artifact has no MTP block")?;
    ensure!(mtp_cursor(cache, slot)? == s, "MTP cursor is not at {s} before drafting");
    let drafted = model.with_gpu_cache_mut(|device, registry| {
        let mut drafts = Vec::with_capacity(depth_cap);
        let mut chain_hidden: Option<MlxBuffer> = None;
        let mut chain_token = sequence[s];
        for depth in 0..depth_cap {
            let shared_embed = model.embed_tokens_gpu_in_context(&[chain_token], device, registry)?;
            let previous = chain_hidden.as_ref().unwrap_or(verifier_hidden);
            let (token, next_hidden) = mtp.forward_draft_greedy_for_token(
                previous,
                chain_token,
                &shared_embed,
                cache,
                slot,
                &[(s + depth) as i32; 4],
                device,
                registry,
                &model.cfg,
            )?;
            drafts.push(token);
            if token != sequence[s + depth + 1] {
                break;
            }
            chain_token = token;
            chain_hidden = Some(next_hidden);
        }
        Ok(drafts)
    });
    cache.truncate_mtp_to_for_slot(slot, s as u32)?;
    drafted
}

fn probe_agent(
    model: &Qwen35Model,
    transcript: &AgentTranscript,
    completion_chunk: usize,
) -> Result<Vec<TurnProbe>> {
    let sequence = &transcript.sequence;
    let hidden_size = model.cfg.hidden_size;
    let slot = SlotId(0);
    let capacity = u32::try_from((sequence.len() + 16).next_multiple_of(256))?;
    model.ensure_gpu_cache_primed()?;
    let mut cache = model.with_gpu_cache_mut(|device, _registry| {
        HybridKvCache::new_with_options(
            &model.cfg,
            device,
            capacity,
            1,
            crate::serve::api::tq_packed_descriptor::is_tq_active_mode(),
        )
    })?;
    ensure!(cache.mtp_slot.is_some(), "cache has no MTP slot");

    let mut probes = Vec::with_capacity(transcript.turns.len());
    let mut cursor = 0usize;
    // `h_nextn` of position `cursor - 1`.
    let mut pending: Option<MlxBuffer> = None;
    for span in &transcript.turns {
        let mut probe = TurnProbe {
            runs: Vec::with_capacity(span.completion_len.saturating_sub(1)),
            drafts: Vec::with_capacity(span.completion_len.saturating_sub(1)),
            argmax_rows: 0,
            argmax_agree: 0,
        };
        // Prompt segment: the server's prefill call and prompt catch-up.
        ensure!(cursor < span.prompt_len, "turn {} has no new prompt tokens", span.turn);
        while cursor < span.prompt_len {
            let end = (cursor + PROMPT_CHUNK).min(span.prompt_len);
            let tokens = &sequence[cursor..end];
            let positions = positions_for_range(cursor as i32, tokens.len());
            let (logits, hidden) =
                model.forward_gpu_last_logits_with_hidden(tokens, &positions, &mut cache, slot)?;
            mtp_catch_up(model, &mut cache, slot, tokens, cursor, pending.as_ref(), &hidden)?;
            pending = Some(last_hidden_row(&hidden, hidden_size)?);
            if end == span.prompt_len {
                probe.argmax_rows += 1;
                probe.argmax_agree += u64::from(argmax(&logits) == sequence[end]);
            }
            cursor = end;
        }
        // Completion segment: verifier rows, then per position draft and
        // one-row catch-up.
        let terminal = span.end();
        while cursor <= terminal {
            let end = (cursor + completion_chunk).min(terminal + 1);
            let tokens = &sequence[cursor..end];
            let positions = positions_for_range(cursor as i32, tokens.len());
            let (logits, hidden) =
                model.forward_gpu_with_nextn_hidden_buffer(tokens, &positions, &mut cache, slot)?;
            let logits = logits
                .as_slice::<f32>()
                .map_err(|error| anyhow!("logits view: {error}"))?;
            ensure!(logits.len() % tokens.len() == 0, "logits rows");
            let vocab = logits.len() / tokens.len();
            for row in 0..tokens.len() {
                let s = cursor + row;
                if s < terminal {
                    probe.argmax_rows += 1;
                    probe.argmax_agree += u64::from(
                        argmax(&logits[row * vocab..(row + 1) * vocab]) == sequence[s + 1],
                    );
                    let verifier_hidden = pending.as_ref().context("verifier hidden")?;
                    let drafts = draft_chain(
                        model,
                        &mut cache,
                        slot,
                        sequence,
                        s,
                        MAX_DEPTH.min(terminal - s),
                        verifier_hidden,
                    )?;
                    let run = drafts
                        .iter()
                        .zip(&sequence[s + 1..])
                        .take_while(|(draft, committed)| draft == committed)
                        .count();
                    probe.runs.push(run as u8);
                    probe.drafts.push(drafts);
                }
                let row_hidden = nth_hidden_row(&hidden, hidden_size, row as u64)?;
                mtp_catch_up(
                    model,
                    &mut cache,
                    slot,
                    &sequence[s..=s],
                    s,
                    pending.as_ref(),
                    &row_hidden,
                )?;
                pending = Some(row_hidden);
            }
            cursor = end;
        }
        probes.push(probe);
    }
    Ok(probes)
}

fn run_probe() -> Result<Value> {
    let gguf = env_path(GGUF_ENV);
    let run = env_path(RUN_ENV);
    let out = env_path(OUT_ENV);
    let completion_chunk = std::env::var(COMPLETION_CHUNK_ENV)
        .ok()
        .map(|value| value.parse::<usize>())
        .transpose()?
        .unwrap_or(DEFAULT_COMPLETION_CHUNK);
    ensure!(completion_chunk > 0, "{COMPLETION_CHUNK_ENV} must be positive");

    let renderer = ServingRenderer::from_gguf(&gguf)?;
    let transcripts = reconstruct_run(&renderer, &run)?;
    let transcript = transcript_evidence(&renderer, &run, &transcripts);
    drop(renderer);

    let load_started = Instant::now();
    let model = load_serving_model(&gguf)?;
    ensure!(model.mtp.is_some(), "{} has no MTP block", gguf.display());
    model.ensure_gpu_cache_primed()?;
    let load_seconds = load_started.elapsed().as_secs_f64();

    let mut agents = Vec::new();
    let mut all_runs: Vec<Vec<u8>> = Vec::new();
    let (mut argmax_rows, mut argmax_agree) = (0u64, 0u64);
    let probe_started = Instant::now();
    for agent in &transcripts {
        let started = Instant::now();
        let probes = probe_agent(&model, agent, completion_chunk)
            .with_context(|| format!("probe {}", agent.agent))?;
        let agent_runs = probes.iter().map(|probe| probe.runs.clone()).collect::<Vec<_>>();
        eprintln!(
            "mtp-probe {}: {} turns, {:.1}s, {}",
            agent.agent,
            probes.len(),
            started.elapsed().as_secs_f64(),
            round_summary(&agent_runs)["by_n_max"]
        );
        agents.push(json!({
            "agent": agent.agent,
            "seconds": started.elapsed().as_secs_f64(),
            "rounds": round_summary(&agent_runs),
            "turns": agent.turns.iter().zip(&probes).map(|(span, probe)| json!({
                "turn": span.turn,
                "prompt_tokens": span.prompt_len,
                "decision_tokens": span.completion_len,
                "accept_runs": probe.runs,
                "drafts": probe.drafts,
                "target_argmax_rows": probe.argmax_rows,
                "target_argmax_agree": probe.argmax_agree,
            })).collect::<Vec<_>>(),
        }));
        for probe in &probes {
            argmax_rows += probe.argmax_rows;
            argmax_agree += probe.argmax_agree;
        }
        all_runs.extend(agent_runs);
    }

    let report = json!({
        "schema": SCHEMA,
        "gguf": gguf.display().to_string(),
        "max_depth": MAX_DEPTH,
        "prompt_chunk": PROMPT_CHUNK,
        "completion_chunk": completion_chunk,
        "tq_kv_active": crate::serve::api::tq_packed_descriptor::is_tq_active_mode(),
        "load_seconds": load_seconds,
        "probe_seconds": probe_started.elapsed().as_secs_f64(),
        "transcript": transcript,
        "target_argmax_agreement": {
            "rows": argmax_rows,
            "agree": argmax_agree,
            "rate": (argmax_rows > 0).then(|| argmax_agree as f64 / argmax_rows as f64),
            "note": "teacher-forced target argmax without the served repetition penalty vs the committed token",
        },
        "rounds": round_summary(&all_runs),
        "agents": agents,
    });
    std::fs::write(&out, serde_json::to_vec_pretty(&report)?)
        .with_context(|| format!("write {}", out.display()))?;
    Ok(report)
}

#[test]
#[ignore = "needs a recorded run and the GGUF metadata; CPU only"]
fn mtp_probe_transcripts_reconstruct() {
    let gguf = env_path(GGUF_ENV);
    let run = env_path(RUN_ENV);
    let renderer = ServingRenderer::from_gguf(&gguf).expect("serving renderer");
    let transcripts = reconstruct_run(&renderer, &run).expect("reconstruct every agent");
    for agent in &transcripts {
        for span in &agent.turns {
            println!(
                "{} turn {}: request {} prompt {} completion {} decisions {} ({})",
                agent.agent,
                span.turn,
                span.request_sha256,
                span.prompt_len,
                span.usage_completion_tokens,
                span.completion_len,
                span.completion_proof
            );
        }
    }
    let evidence = transcript_evidence(&renderer, &run, &transcripts);
    println!(
        "{}",
        serde_json::to_string_pretty(&evidence).expect("serialize evidence")
    );
}

#[test]
#[ignore = "needs a recorded run and a Qwen MTP GGUF on Apple GPU"]
fn mtp_acceptance_probe_real_run() {
    let report = run_probe().expect("MTP acceptance probe");
    println!(
        "{}",
        serde_json::to_string_pretty(&json!({
            "rounds": report["rounds"],
            "target_argmax_agreement": report["target_argmax_agreement"],
            "load_seconds": report["load_seconds"],
            "probe_seconds": report["probe_seconds"],
        }))
        .expect("serialize summary")
    );
}

#[test]
fn round_tally_follows_accept_runs_per_n_max() {
    // Six round-start positions before the terminal token; each run is
    // capped by the tokens left (positions - index).
    let runs = [3u8, 2, 1, 0, 2, 1];
    let tally = |n_max| {
        let mut tally = RoundTally::default();
        tally.add_turn(&runs, n_max);
        tally
    };
    assert_eq!(
        tally(3),
        RoundTally {
            rounds: 2,
            emitted: 6,
            accepted: 5,
            drafted: 6,
            histogram: [0, 0, 1, 1],
        }
    );
    assert_eq!(
        tally(2),
        RoundTally {
            rounds: 3,
            emitted: 6,
            accepted: 4,
            drafted: 6,
            histogram: [1, 0, 2, 0],
        }
    );
    assert_eq!(
        tally(1),
        RoundTally {
            rounds: 3,
            emitted: 6,
            accepted: 3,
            drafted: 3,
            histogram: [0, 3, 0, 0],
        }
    );
    let depths = depth_acceptance(&[runs.to_vec()]);
    // Depth 1: all six positions eligible, five accept.
    assert_eq!(depths[0]["eligible"], 6);
    assert_eq!(depths[0]["accepted"], 5);
    // Depth 2: positions with run >= 1 and >= 2 tokens left: 0, 1, 2, 4;
    // runs >= 2 at 0, 1, 4.
    assert_eq!(depths[1]["eligible"], 4);
    assert_eq!(depths[1]["accepted"], 3);
    // Depth 3: run >= 2 and >= 3 tokens left: 0, 1.
    assert_eq!(depths[2]["eligible"], 2);
    assert_eq!(depths[2]["accepted"], 1);
}

#[test]
fn single_position_turn_has_no_rounds() {
    let mut tally = RoundTally::default();
    tally.add_turn(&[], 3);
    assert_eq!(tally, RoundTally::default());
}
