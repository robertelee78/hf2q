# ADR-063 Q4-QWEN36: Qwen3.6 harness against Qwen's recommended agentic setup

Status: prepared, 2026-11-03. No campaign has run yet. This record holds the setup diff with sources, the
third-party diagnostic GGUFs, and the paired campaigns that separate harness, conversion and model causes
(story #662). The starting point is
[adr063-qwen36-task-failure-diagnosis.md](adr063-qwen36-task-failure-diagnosis.md): canonical workload,
hf2q 5/24 against the peer 4/24 (p = 0.77), BelowFloor, attributed to the weights.

## Sources

- Model card: `Qwen/Qwen3.6-35B-A3B` at `995ad96eacd98c81ed38be0c5b274b04031597b0`. Hashes:
  - `README.md`: sha256 `c4ddaa06…`
  - `generation_config.json`: `e70c136c…`
  - `chat_template.jinja`: `e84f32a2…`

  These are the files bound by the hf2q conversion receipt. README line numbers are cited as `card:N`, and
  template lines as `tpl:N`.
- Qwen-Agent at `QwenLM/Qwen-Agent@31a4d36d123688581a9e9744427272b33ce940e0`. The card recommends it for
  agentic use (card:870-946).
- Harness at origin/main `d817dffa`. Files: `scripts/four_agent/qwen38_n1_smoke.py`, `qwen38_n1_verdict.py`,
  `fixtures/canonical-coding-requests.json`, `q5_cases/qwen36.json` and `peer-qwen36.json`, and
  `scripts/serve_qwen36_opencode.sh`.

Copies of every vendor file, and the comparison scripts, are in the evidence directory
`~/.local/state/hf2q/v0130/q4-qwen36/`.

## Setup diff

The first column is Qwen's recommendation, the second what the canonical harness sends. "Vendor arm" says
what `--request-setup qwen36-vendor-coding-v1` changes.

| Item | Qwen recommendation (source) | Canonical harness (source) | Difference | Vendor arm |
|---|---|---|---|---|
| Thinking mode | On by default; the generation prompt opens `<think>` (card:670, tpl:147-153). | On. Neither engine sends `enable_thinking`, so the template default applies. The peer runs `--reasoning on`. | None | Same |
| Past reasoning re-sent | Qwen-Agent re-sends `reasoning_content` in every assistant message (`base.py:430-431`). The template renders the reasoning of every assistant turn after the last user query (tpl:100). | The client appends the server's assistant message verbatim, including `reasoning_content` (`qwen38_n1_smoke.py`; the `SSE`/`unary` parsers keep the field). | None | Same |
| `preserve_thinking` | The Qwen-Agent example sets it true (card:896-897, 916). It renders the reasoning of turns before the last user query (card:836-867, tpl:100). | True on both engines: hf2q's default (`handlers.rs:1736-1738`) and the peer's `--chat-template-kwargs`. The task has one user query, so every assistant turn is already after `last_query_index` and the flag changes no token. | None | Same |
| Sampling | Thinking mode for precise coding: temperature 0.6, top_p 0.95, top_k 20, min_p 0, presence 0 (card:1006-1007). General thinking: temperature 1.0, presence 1.5 (card:1004-1005). `generation_config.json`: temperature 1.0, top_k 20, top_p 0.95. Agent benchmarks: temperature 1.0, top_p 0.95 (card:307-310). | Greedy. The request omits sampling, and hf2q resolves temperature 0, top_p 1, top_k 0 (`SETTINGS.md:18`). The peer runs `--temp 0 --top-k 0 --top-p 1 --min-p 0`. | **Different** | Precise-coding set on both engines |
| Repetition penalty | 1.0 (card:1005-1009) | 1.05. hf2q applies it to generated tokens; the peer applies it over a 64-token window that includes the prompt (`SETTINGS.md:42`). | **Different** | 1.0 on both |
| Presence penalty | 0 for precise coding, 1.5 for general use | 0 (omitted). hf2q accepts the field but its sampler does not consume it (`engine.rs` `SamplingParams` docs). | None for precise coding. The general set is not reproducible on hf2q. | 0 |
| Thinking budget | None (no cap in any vendor example) | hf2q: 2048 on a fresh turn and 512 on a tool continuation, closed with `\nI need to answer now.</think>` (`serve_qwen36_opencode.sh`, `qwen_thinking_policy.rs`). The peer gets `reasoning_budget_tokens` 2048/512 per request. The cap was reached in 2 of 454 hf2q turns in the canonical campaign. | **Different** | No budget on either engine (hf2q `THINKING_TOKEN_BUDGET=0`, `TOOL_THINKING_TOKEN_BUDGET=0`; peer `--reasoning-budget -1`, no request budget) |
| Output allowance | 32768, or 81920 for hard benchmarks (card:1012) | `max_completion_tokens` 8192 | Different, **kept** | 8192. The largest completion across about 1,850 Qwen3.6 campaign turns (three campaigns) was 724 tokens, an 11× margin. A `length` finish fails the run visibly. |
| System prompt | None required. Qwen-Agent's default is empty (`schema.py:19`). The template appends any system text after the tools block (tpl:54-59). | Fixture: "You are editing a small Rust library. Use read_file, write_file and run_tests. Preserve unrelated changes.", prefixed by the per-agent identity line | Task content, **kept** | Same |
| Tool-call format | The template's native XML: `<tool_call><function=…><parameter=…>` (tpl:53). Parser `qwen3_coder` in the card's vLLM and SGLang commands (card:579, 611). | The same template bytes on both engines (`e84f32a2…`; the peer runs `--chat-template-file`). Each engine parses the native XML. Both constrain tool calls with a lazy grammar (preflight `auto_lazy_tool_grammar_charged`), which the card's commands do not. | Constrained decoding, **kept** | Same |
| Tool schema rendering | `tool \| tojson` in client insertion order | hf2q keeps insertion order (`serde_json` and `minijinja` `preserve_order`, `Cargo.toml`) and drops `strict: false` (`schema.rs`). The peer ignores `strict` too. Every tool description is the generic "Operate on the isolated coding fixture." | Harness choice; no vendor guidance | Same |
| Agent loop and repair | Qwen-Agent loops up to 20 model calls per run (`settings.py:24`). The vendor's coding benchmarks are multi-turn scaffolds in which the model sees test output and edits again (card:307-311). | A strict six-step protocol with exactly one call per response: read, read, write, `run_tests`, read, `run_tests`, then a summary. There is no repair turn (`qwen38_n1_verdict.py` `EXPECTED`, `qualify`). | **Largest difference.** Every failing agent in the diagnosis writes a fix after E0282 but cannot submit it. | Unchanged. Changing it changes the canonical workload, which is an operator decision. |
| Context | At least 128K (card:557) | 262144 per slot | None | Same |
| Weights | The vendor serves BF16/FP8 | Q4_K_M (hf2q recipe). The diagnosis's BF16 reference produced the idiom in 41/48 replays. | Quantization | Third-party GGUF arms |
| Transport | Streaming (Qwen-Agent `use_raw_api` requires it, `base.py:222`) | Unary and SSE alternate by turn; the prompt is identical either way | None for tokens | Same |

Thinking mode, reasoning re-send, `preserve_thinking`, tool-call format and context already match. The
harness differs from the recommended setup in four places: sampling (including the penalty), the thinking
budget, the output allowance and the strict workflow. The vendor arm closes the first two. The output
allowance is kept with the evidence above. The workflow is the canonical workload and stays.

The unsloth GGUF embeds a modified template (`55d49314…`: developer role, merged system messages, a
mapping check on tool arguments). `template_render_check.py` renders both templates over all 1,931
transcript prefixes of the 192 agents in the canonical campaign. All 1,931 renders are identical (Jinja2
reference renderer, arguments as mappings). hf2q serves the GGUF-embedded template; the peer serves the
vendor template file on every GGUF.

## Third-party diagnostic GGUFs

These files are diagnostic data only, under `~/.local/share/hf2q/diagnostic/qwen36/`. They are never a
conversion substitute or a release artifact. The receipt is `download-receipts.json` in the evidence
directory. Both sha256 values equal the Hub LFS object ids.

| GGUF | Repository @ revision | sha256 | Bytes | Recipe (observed from the GGUF directory) | Template |
|---|---|---|---:|---|---|
| ggml-org Q4_K_M | `ggml-org/Qwen3.6-35B-A3B-GGUF@7878dc9a` | `671e47e0…40c7` | 20,419,565,568 | Source `995ad96e` (`.src_sha`), no imatrix, no MTP layer. Attention, SSM projections and shared experts Q8_0; routed experts Q4_K; output Q6_K. | `e84f32a2…` (vendor) |
| unsloth UD-Q4_K_M | `unsloth/Qwen3.6-35B-A3B-GGUF@a483e9e6` | `ac0e2c11…1a61` | 22,134,528,992 | Unsloth Dynamic with imatrix (the repository has no plain Q4_K_M). Attention and shared experts Q8_0; `ssm_alpha`/`ssm_beta` F32; routed down Q5_K/Q6_K, routed up/gate Q4_K; `token_embd` Q8_0; no MTP layer. | `55d49314…` |
| hf2q Q4_K_M (ours) | converted by hf2q 0.1.27 @`5d1ffe7f` from `995ad96e` | `815a22ca…8220` | 21,712,410,208 | Attention and SSM projections mostly Q4_K (`attn_qkv`/`attn_v` partly Q6_K); routed down Q4_K/Q6_K; output Q6_K; MTP layer kept. | `e84f32a2…` |

Neither third-party file is a stock Q4_K_M: both keep attention and shared experts at Q8_0, where ours uses
Q4_K. They therefore also test "more precision outside the routed experts". `hf2q info` (static
preflight, Metal not initialized) reports both as serve-ready.

## Harness change

`--request-setup qwen36-vendor-coding-v1` (`request_setup.py`) sends the precise-coding sampling
explicitly on every request, with the same values on hf2q and the peer. The cases are
`q5_cases/qwen36-vendor.json` and `peer-qwen36-vendor.json`. The approval binds the setup, the agent and
coordinator summaries record it, and `aggregate` and the Q2 run audit replay it. The README section is
"Qwen3.6 vendor request setup".

## Campaigns

Each campaign is the 24-instance paired task campaign (`quality_campaign.py`, Q1 policy §3.5): 48 Q5 runs
in ABBA order. All campaigns use one hf2q binary, built from the harness commit, and the e86f5fd1 peer
build of the diagnosis. Commands are in `campaigns.sh` in the evidence directory; `campaigns.sh dry-run`
validates all six plans with no missing variables.

| Id | Setup | GGUF | Arms | Role |
|---|---|---|---|---|
| C0 | Canonical | Ours | hf2q / peer | Control: the 5/24 vs 4/24 baseline on the current binary |
| C1 | Vendor | Ours | hf2q / peer | Harness: the recommended setup |
| C2 | Canonical | ggml-org | hf2q / peer | Conversion |
| C3 | Canonical | unsloth | hf2q / peer | Conversion |
| C4 | Vendor | ggml-org | hf2q / peer | Optional; run only if C1 moves the pass rate |
| C5 | Vendor | unsloth | hf2q / peer | Optional; run only if C1 moves the pass rate |

How each question is answered (same instance ids in every campaign; "per arm" means the same engine):

- **Engine.** Within each campaign, the paired verdict. A hf2q `Loss` (one-sided sign p < 0.05) is a hf2q
  defect under that setup or GGUF.
- **Harness.** C1 against C0 per arm, with an exact sign test on discordant instances. If the vendor setup
  reaches the floor (at least 18/24) or beats C0 at p < 0.05 on both engines, the setup is causal.
- **Conversion.** C2 and C3 against C0 per arm. A third-party GGUF that passes significantly more on both
  engines implicates the hf2q Q4_K_M recipe, most likely the Q4_K attention and SSM projections, since
  both third-party files keep those at Q8_0. Agreement with C0 keeps the conversion excluded.
- **Model.** If every arm stays BelowFloor under both setups and all three GGUFs, the failure is the model
  on the strict workload. Q4-QWEN36 then returns to the operator as `needs_decision`. The candidate
  decision is a workload that allows a repair turn, the largest remaining difference in the table.

Before C1, run one Q5 calibration of each vendor case on the canonical task. This proves the sampled
four-slot path and the peer's settings outside the campaign, where an engine error would otherwise score
as a task failure.

## Run budget

The canonical paired campaign took 29.0 min for 48 runs. Median run wall time was 38.5 s on hf2q (client
30.6 s) and 29.5 s on the peer. These estimates are inferences from that campaign:

- C0, C2, C3: about 30-35 min each.
- C1: up to about 60 min. Sampled requests leave hf2q's GPU greedy path.
- The two vendor-case Q5 smokes: about 2 min.
- The hf2q release build: CPU only, under the host lock.

The core set (C0-C3 plus the smokes) is about 2.5-3 h of GPU lane. C4 and C5 add up to 2 h. Each campaign
writes about 70 MB of evidence. The two downloads use 42.6 GB.
