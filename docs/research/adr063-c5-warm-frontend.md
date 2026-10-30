# ADR-063 C5: warm-request frontend overhead (#361)

Status: deciding experiment complete (2026-10-30). The hypothesis is falsified for all four families, so no acceleration path is implemented. The verdict was independently reviewed.

## Question and gate

C5 asks whether warm four-agent requests spend enough time in rendering, tokenization and prefix lookup to justify an exact reuse or acceleration path. The ADR gate: open an implementation only if frontend work exceeds **10% of measured warm semantic TTFT** and an exact path has credible end-to-end headroom. Otherwise, close the deciding experiment with evidence.

## What the handler does per request (source, this branch)

`prepare_chat_generation_core` in `src/serve/api/handlers.rs` runs on the request task before the worker sees the request:

| Step | Families | Work |
|---|---|---|
| Pool admission | all | `resolve_engine_for_request` (warm: pool hit plus a lease) |
| Constraint compile | all | output constraint and tool grammar (only when requested) |
| Primary render | all | minijinja render of the full transcript. DeepSeek-V4 uses the Rust encoder port, not Jinja |
| Primary tokenize | all | `tokenizers` encode of the full rendered prompt |
| Stable-prefix render | Gemma 4, Qwen 3.5+ | the same transcript without the generation cue |
| Stable-prefix tokenize | Gemma 4, Qwen 3.5+ | second full encode |
| Prefix match | Gemma 4, Qwen 3.5+ | `starts_with` boundary check and the Gemma full-input tag |
| Probe render | registered families | a one-message render that checks for an open reasoning block |
| Budget markers | all | encodes the short reasoning end/close marker strings |

The template is re-parsed on each render, and the whole conversation is re-tokenized on every turn. Slot affinity, restore and suffix prefill run later, on the engine worker, and are not frontend work.

## Existing evidence (E = `/Users/robert/.local/state/hf2q/campaigns/adr063-preflight-20261023`)

- The server logs have no timestamps: the text subscriber uses `.without_time()`. The `chat completion request received` → `prepared; dispatching` window therefore cannot be measured from `server.stderr`. The F2 native trace (`$E/f2-qwen38-native-trace-operating90-20261028T010639Z/native-phase-trace.jsonl`) starts at worker admission, so it does not cover the handler either. No existing artifact measures frontend time directly.
- Warm prompt sizes in the four-agent workload are small. Qwen3.8 traced run: 728–2368 prompt tokens per request. DeepSeek-V4 joint run `ds4-joint-ab-v1/b1-*`: 757–2323.
- Warm semantic TTFT (F2 scoreboard, median of streamed turns):

| Family | Warm TTFT | Gate (10%) |
|---|---|---|
| Qwen3.8 | 8.65–8.74 s | ≈ 870 ms |
| DeepSeek-V4 | 9.65–9.99 s | ≈ 965 ms |
| DeepSeek-V4, engine-side `first semantic stream event` (joint B arm, warm turns) | 1.15–3.4 s | ≥ 115 ms |
| Gemma 4 (full-input checkpoint on) | 0.768 s | ≈ 77 ms |

### Offline CPU estimate (descriptive, not hardware evidence for the served path)

Method: the same libraries hf2q links, through their Python bindings, on this host's CPU. `tokenizers` 0.22.2 (the crate is 0.22) and `minijinja` (Rust minijinja). Inputs were the families' published `tokenizer.json` and `chat_template.jinja` from the local HF cache, plus the captured OpenCode request `$E/client-capture/requests.json` (12 tools) extended with N synthetic `read` tool-call/result turns. Each value is the median of 10.

| Family | Tokens | Template parse+render | Tokenize | Per request: 3 renders + 2 encodes |
|---|---|---|---|---|
| Qwen3.8 | 5,259 | 0.47 ms | 3.35 ms | 8.1 ms |
| Qwen3.8 | 26,209 | 1.38 ms | 15.4 ms | 34.9 ms |
| Gemma 4 | 5,046 | 0.72 ms | 3.01 ms | 8.2 ms |
| Gemma 4 | 25,399 | 8.25 ms | 10.5 ms | 45.7 ms |
| DeepSeek-V4 (tokenize only) | 5,515 | n/a (Rust encoder) | 3.48 ms | — |

Template parsing costs ≤ 0.15 ms of each render, so caching the compiled template has no headroom. Tokenization costs about 0.6 µs per token and dominates. The last column overstates the cost, because the probe render covers one message, not the transcript.

[INFERENCE] For the four-agent workload (≤ 2.4k tokens), handler frontend work is about 3–5 ms per request on Qwen3.8 and Gemma 4, and about 2 ms on DeepSeek-V4. Shares of warm TTFT:

- Qwen3.8: about 0.05%
- DeepSeek-V4: about 0.02% against the client TTFT, and below 0.2% against the engine-side TTFT
- Gemma 4: about 0.5%

Even a 26k-token transcript at an unchanged TTFT stays under the gate: Gemma 4 at about 6% is the worst case. Every family is one to three orders of magnitude below the 10% gate.

### Candidate exact paths (not implemented)

- **Incremental stable-prefix tokenization.** The stable prefix is a text prefix of the primary prompt. Encoding only the suffix and verifying the boundary would save one full encode (about 1.5 ms at 2.4k tokens). An exact version must re-encode a back-off window across the boundary, because BPE merges and pre-tokenizer splits cross it. It must also handle special tokens, template changes and vision placeholder expansion. It must match the full encode for every prefix/suffix split in the family fixtures. The saving is below 0.1% of TTFT, so it fails the headroom condition.
- **Cross-turn incremental tokenization.** This means reusing the previous turn's ids and encoding only the appended messages. It needs per-conversation state in the frontend, which conflicts with conversation isolation. Qwen also re-renders historical turns when `last_query_index` moves. Its headroom is the same few milliseconds.
- **Compiled-template cache.** Measured above: no headroom.

## Instrumentation added (this branch)

`HF2Q_FRONTEND_TIMING=1` (read once per process; any other value, or unset, means off) makes every dispatched chat completion emit one INFO line. The target is `hf2q::serve::frontend_timing` and the message is `frontend timing`. Source: `src/serve/api/frontend_timing.rs`.

| Field | Meaning |
|---|---|
| `seq`, `family`, `stream`, `outcome` | Process-local sequence, `ArchFamily`, stream flag. `outcome` is `first_reasoning`, `first_content` or `first_tool_call` (stream); `done_without_output`, `error` or `closed_before_output` (stream); `ok` or `error` (unary) |
| `messages`, `tools`, `rendered_bytes`, `prompt_tokens`, `stable_prefix_tokens` | Request shape and token counts. `stable_prefix_tokens` is absent when no boundary is accepted |
| `admission_us` | Pool resolution and model lease |
| `vision_us` | Vision preflight (≈ 0 for text-only requests) |
| `constraint_us` | Output constraint and tool grammar compile |
| `render_us`, `tokenize_us` | Primary render and encode |
| `overflow_rewrite_us` | Truncate/summarize rewrite (0 unless the prompt overflowed) |
| `stable_render_us`, `stable_tokenize_us`, `prefix_match_us` | Stable-prefix render, encode and match |
| `probe_render_us` | Reasoning-open probe render |
| `budget_us` | Thinking-budget resolution, including marker encodes |
| `enqueue_us` | Stream budget admission plus worker enqueue (0 for unary) |
| `prepare_us` | Request receipt to prepared |
| `first_output_us` | Stream only: request receipt to the first reasoning, content or tool-call event, or the terminal event |
| `completion_us` | Unary only: request receipt to worker result |

When the gate is off, each span is a branch on `None`: no clock read, no allocation and no stream relay. When on, streams pass through one extra relay task that forwards every event unchanged and in order. Client disconnect still reaches the worker on its next send.

`frontend = constraint + render + tokenize + overflow_rewrite + stable_render + stable_tokenize + prefix_match + probe_render + budget`. Admission, vision and enqueue are reported but are not counted. `first_output_us - prepare_us - enqueue_us` is the worker-side residual: affinity wait, restore, suffix prefill and first semantic output.

## Root commands

Focused tests (root):

```sh
cargo test --locked --bin hf2q frontend_timing
cargo test --locked --bin hf2q a5d_handler_429
python3 -m unittest discover -s scripts/four_agent -p 'test_c5_frontend_summary.py' -v
cargo build --release --locked
```

Hardware: one four-agent run per family with the gate on. Use the checked-in Q5 cases, changing only the server env. Q5 environment, approvals and the negative-control receipt are as in `scripts/four_agent/README.md` → Q5 → Root commands. Run one family at a time, with no other runtime loaded.

```sh
E=/Users/robert/.local/state/hf2q/campaigns/adr063-preflight-20261023
C5="$E/c5-frontend-timing-$(date -u +%Y%m%dT%H%M%SZ)"; mkdir -p "$C5"
export HF2Q_Q5_HF2Q_BIN=/opt/hf2q-worktrees/perf-c5-warm-frontend/target/release/hf2q
export HF2Q_Q5_CARGO=/abs/cargo HF2Q_Q5_TOOLCHAIN=/abs/toolchain
export HF2Q_Q5_SANDBOX_PROOF=/abs/sandbox-proof.json HF2Q_Q5_SANDBOX_PROOF_SHA256=<sha256>

for fam in qwen38 gemma4 deepseek4 qwen36; do
  python3 - "$fam" "$C5" <<'PY'
import json, sys
fam, out = sys.argv[1:]
case = json.load(open(f"scripts/four_agent/q5_cases/{fam}.json"))
case["serve"]["env"]["HF2Q_FRONTEND_TIMING"] = "1"
case["notes"].append("ADR-063 C5: checked-in case plus serve.env HF2Q_FRONTEND_TIMING=1; nothing else changed.")
json.dump(case, open(f"{out}/{fam}-case.json", "w"), indent=2)
PY
done

# Qwen3.8 27B
HF2Q_Q5_MODEL=/abs/Qwen3.8-27B-vanilla.gguf HF2Q_Q5_CLIENT_APPROVAL_TEMPLATE=/abs/qwen38-client-approval.json \
  python3 scripts/four_agent/q5_calibration.py "$C5/qwen38-case.json" \
  --negative-control-receipt "$NC_RECEIPT" --output "$C5/qwen38"
# Gemma4 26B-A4B
HF2Q_Q5_MODEL=/abs/gemma-4-26B-A4B-it-hf2q-q4_k_m.gguf HF2Q_Q5_CLIENT_APPROVAL_TEMPLATE=/abs/gemma4-client-approval.json \
  python3 scripts/four_agent/q5_calibration.py "$C5/gemma4-case.json" \
  --negative-control-receipt "$NC_RECEIPT" --output "$C5/gemma4"
# DeepSeek-V4 Flash
HF2Q_Q5_MODEL=/abs/DeepSeek-V4-Flash-0731-hf2q-deepseek4-agentic-q2.gguf \
HF2Q_Q5_CLIENT_APPROVAL_TEMPLATE=/abs/deepseek4-client-approval.json \
  python3 scripts/four_agent/q5_calibration.py "$C5/deepseek4-case.json" \
  --negative-control-receipt "$NC_RECEIPT" --output "$C5/deepseek4"
# Qwen3.6 35B-A3B (optional fourth family)
HF2Q_Q5_MODEL=/abs/Qwen3.6-35B-A3B-vanilla-Q4_K_M.gguf HF2Q_Q5_CLIENT_APPROVAL_TEMPLATE=/abs/qwen36-client-approval.json \
  python3 scripts/four_agent/q5_calibration.py "$C5/qwen36-case.json" \
  --negative-control-receipt "$NC_RECEIPT" --output "$C5/qwen36"

# Decision, per family
for fam in qwen38 gemma4 deepseek4 qwen36; do
  python3 scripts/four_agent/c5_frontend_summary.py "$C5/$fam/server.stderr" > "$C5/$fam-c5-summary.json"
done
```

All four Q5 cases already set `RUST_LOG=info` or launch `hf2q -v serve`, so the INFO line is kept. In `server.stderr`, check that each family logs one `frontend timing` line per `chat completion prepared` line.

## Decision rule and what closes C5

For each family, `c5_frontend_summary.py` takes warm streamed requests: `stream=true`, more than 2 messages and a `first_output_us`. It reports the median of `frontend_us / first_output_us`.

- **Falsified (expected):** the median share is ≤ 10% for every family. C5 then closes with the summary JSONs, the Q5 receipts and the paired client `semantic_ttft_seconds` (from `client/summary.json`) as a cross-check on `first_output_us`. No implementation follows.
- **Material:** the median share is > 10% for some family. Open the implementation for that family only, starting with the span that dominates (expected: `tokenize_us` + `stable_tokenize_us`). Gate it with an oracle: token ids identical to a full `tokenizers` encode for every prefix/suffix split of each family template's rendered transcripts. Use `scripts/four_agent/fixtures/canonical-coding-requests.json` and the captured agent transcripts under `$E/client-capture*`. Cover randomized boundary edits, special tokens, template switches and image placeholders. Then require a paired end-to-end TTFT A/B against the gate-off binary.

### Measured verdict (2026-10-30, M5 Max 128 GB): falsified

Source `c4316b3f` with `HF2Q_FRONTEND_TIMING=1`, one four-agent coding run per family under the Q5 supervisor. Warm streamed requests:

| family | n | frontend p50 | first output p50 | share p50 | share max | client semantic TTFT p50 (range) | verdict |
|---|---:|---:|---:|---:|---:|---:|---|
| DeepSeek-V4 | 12 | 2.30 ms | 3.32 s | 0.074% | 0.122% | 9.84 s (8.84–11.44) | not material |
| Qwen3.8 | 12 | 2.89 ms | 1.86 s | 0.156% | 0.297% | 9.72 s (5.35–24.79) | not material |
| Gemma 4 | 12 | 1.94 ms | 1.26 s | 0.156% | 0.583% | 1.54 s (0.64–2.82) | not material |
| Qwen3.6 | 11 | 2.64 ms | 0.555 s | 0.475% | 1.04% | 1.68 s (1.32–7.07) | not material |

DeepSeek-V4, Qwen3.8 and Gemma 4 completed the protocol with `CalibrationPass`. The Qwen3.6 run ended `ClientFailed` on the known coding-task divergence (the model writes a non-compiling `parse` call; see #461 and the Q1 quality campaign). That failure happened after warm turns, and every dispatched request still logged its frontend line. So the Qwen3.6 frontend measurement is valid, but the run does not count as a calibration pass.

Cross-check: the server's `first_output_us` is time to the first output event, including a reasoning token, so it is never later than the client's semantic TTFT. Each share therefore overstates the frontend fraction of semantic TTFT.

Disposition per family: falsified for DeepSeek-V4, Qwen3.8, Gemma 4 and Qwen3.6. The worst share, 1.04% (Qwen3.6), is about a tenth of the 10% gate. No implementation follows, and the timing spans stay as an opt-in diagnostic. Receipts and summaries: `$E/c5-runs-v1/{deepseek4-20261030T155611Z,qwen38-20261030T155910Z,gemma4-20261030T160447Z,qwen36-20261030T172555Z}/{receipt.json,c5-summary.json}`.

Separate from C5: time spent queued at the worker is a large part of the server's first-output time (median 718 ms on DeepSeek-V4, up to 1.87 s on Gemma 4). It belongs to the scheduler stories, not the frontend.

## Risks

- The offline numbers use Python bindings over the same Rust libraries, on the same CPU. They are an estimate, not the served path, and the hardware run is what decides.
- "Warm" means more than 2 messages. That covers the four-agent client (system + user opens every agent). Other clients may need `--cold-messages`.
- With the gate on, the stream relay adds one channel hop. Cancellation still propagates, delayed until the worker's next send. Throughput claims must come from gate-off runs.
- Only dispatched requests produce a line. Requests rejected during preparation (4xx/503) are not timed.
