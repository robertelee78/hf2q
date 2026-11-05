# ADR-063 Q4-QWEN36: Qwen3.6 harness against Qwen's recommended agentic setup

Status: C3 measured 2026-11-04 (operator decision 2026-11-03: unsloth GGUF only, canonical setup, v0.1.29
release binary). R1, the repair-turn campaign, was measured 2026-11-04: hf2q `Loss` against the peer. C0,
C1, C2, C4 and C5 have not run. This record holds the setup diff with sources, the
third-party diagnostic GGUFs, the paired campaigns that separate harness, conversion and model causes
(story #662), and the C3 result. The starting point is
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

## C3 result: unsloth UD-Q4_K_M, canonical setup

Campaign `$V/q4-qwen36/campaigns/quality-qwen36-unsloth-udq4km-20261104T001837Z`, where `$V` is
`~/.local/state/hf2q/v0130`:
- Manifest sha256 `80cd6bfa…`, plan `5a732c3f…`.
- 48 runs in 29.8 min. Every run has a task outcome: no Invalid run and no retry.
- hf2q: v0.1.29 release binary, sha256 `86b64b02…`.
- Peer: e86f5fd1 build, `4b604952…`.
- Artifact: `ac0e2c11…`.
- Client sandbox proof: `owned-cleanup-sandbox-capability-20261031T222313Z` (`8b286684…`). The 2026-10-26
  proof no longer matches `qwen38_n1_process.py` on main.

| | hf2q | Peer |
|---|---:|---:|
| Instances passed (C3) | 1/24 | 1/24 |
| Instances passed, canonical campaign (our GGUF) | 5/24 | 4/24 |
| Pass only in C3 / pass only in canonical (same instance ids) | 1 / 5 | 1 / 4 |
| Two-sided exact sign test, C3 vs canonical | p = 0.22 | p = 0.38 |
| Agents failing with E0282 on `map_err(\|e\| e.to_string())` | 62 | 53 |
| First `write_file` containing the idiom | 65/82 (79%) | 64/74 (86%) |
| Same, canonical campaign | 46/89 (52%) | 49/87 (56%) |
| Q5 verdicts | 23 ClientFailed, 1 CalibrationPass | 23 ClientFailed, 1 CalibrationPass |

- **Paired verdict:** BelowFloor. hf2q-only failures 1 and peer-only failures 1 give one-sided p = 0.75,
  so there is no engine difference.
- **Failure signature:** the same line as the diagnosis. Every failing agent either compiled the E0282
  idiom, was stopped by the coordinator after a sibling failed, or (1 per engine) was stopped before its
  first test result arrived.
- **Smokes:** the two canonical-task smokes, `q5-qwen36-unsloth-20261104T001640Z` and
  `q5-peer-qwen36-unsloth-20261104T001748Z`, failed the same way.
- **Conversion:** the third-party GGUF is no better than ours on either engine. Both drops are
  non-significant. It keeps attention, SSM and shared experts at Q8_0 and uses an imatrix, yet it writes
  the non-compiling idiom more often. That matches the diagnosis's trend of the idiom rising with precision
  (Q4 22 → Q8 34 → BF16 41 of 48). The hf2q conversion stays excluded.
- **Confound:** the hf2q arm's C3-vs-canonical comparison also changes the binary (`0b959107` → v0.1.29).
  The peer arm uses the same build in both campaigns.

Analysis: `analyze_c3.py` writes `c3-analysis.json` in the evidence directory.

## Repair-turn workload (operator decision 2026-11-03)

Qwen3.6's race workload grants one repair cycle, on the hf2q-converted GGUF (`815a22ca…`), identically on
hf2q and the peer. The harness option is `--repair-turns 1`. Its cases are `q5_cases/qwen36-repair1.json`
and `peer-qwen36-repair1.json`; the README section is "Qwen3.6 repair-turn workload".

**Rule.** After a failed `run_tests`, the agent's next two calls must be one `write_file src/ports.rs` and
one `run_tests` on the same target. The six steps then resume after the failed `run_tests`. A failed
`run_tests` once the cycle is spent fails the agent at once. The prompt, including the workflow text
"Complete exactly these six steps", is unchanged, so the model is not told about the cycle. The recorded
failures show that it writes a fix on its own.

**Prior from the recorded failures (CPU, no new run).** A failed `run_tests ports` is followed by a
turn-4 `write_file` that the strict workflow refused. In each recorded write, the "same form" is
`s.parse().map_err(|e| e.to_string())` with no turbofish and no closure type annotation. Source:
`repair-prior.txt` in the evidence directory.

| Campaign | Engine | Agents with a failed ports test | Turn-4 write recorded | Keeps the same E0282 form |
|---|---|---:|---:|---:|
| Canonical (our GGUF) | hf2q | 43 | 20 | 10 |
| Canonical (our GGUF) | Peer | 48 | 22 | 2 |
| C3 (unsloth) | hf2q | 62 | 29 | 6 |
| C3 (unsloth) | Peer | 53 | 26 | 0 |

- Most repair writes add `parse::<T>()` or `|e: ParseIntError|`. The rest were stopped by the coordinator
  before turn 4.
- On the repair write, hf2q keeps the failing form more often than the peer: 16/49 vs 2/48 pooled, Fisher
  two-sided p = 0.0004. Canonical alone gives p = 0.006, C3 alone p = 0.02.
- Each engine replays its own transcripts, so prefixes differ. This is a pre-registration flag, not an
  engine verdict. The paired repair campaign decides it. If hf2q shows a `Loss` there, the next step is a
  same-prefix fork replay of the repair turn (`$E/qwen36-fork-v1/replay.py`).

### R1 result: repair-turn campaign, our GGUF

Campaign `$V/q4-qwen36/campaigns/quality-qwen36-repair1-ours-20261104T054600Z`, where `$V` is
`~/.local/state/hf2q/v0130`:
- Manifest sha256 `694700be…`, plan `2584208b…`.
- 48 runs. Every run has a task outcome; there are no supervisor stops and no retries. Cleanup was proven
  on every run.
- hf2q: v0.1.29 release binary `86b64b02…`. Peer: e86f5fd1 build `4b604952…`. GGUF: `815a22ca…`.
- An earlier attempt, `…-20261104T011417Z`, halted at run 10 with a Q5 `swap_growth` stop (peer arm, i05).
  The host orchestrator process had grown to 51 GB resident. That attempt is not part of the verdict; its
  10 settled runs show the same pattern.
- The repair-turn smokes, `q5-qwen36-repair1-20261104T011222Z` and
  `q5-peer-qwen36-repair1-20261104T011222Z`, were CalibrationPass with 4/4 agents on both engines.

| | hf2q | Peer |
|---|---:|---:|
| Instances passed | 14/24 | 23/24 |
| Paired verdict | **Loss** (hf2q-only failures 10, peer-only 1, one-sided p = 0.0059) | |
| Instances passed, canonical strict workload (same ids) | 5/24 | 4/24 |
| Repair vs canonical, two-sided sign test | 9 gained / 0 lost, p = 0.0039 | 19 gained / 0 lost, p = 4e-6 |
| Agents whose first `run_tests` failed | 45/96 | 57/96 |
| Repairs used | 43 | 57 |
| Repair compiled and its test passed | 26 | 56 |
| Repair test failed (agent fails) | 2 | 0 |
| Repair write byte-identical to the agent's failing first write | 16/43 | 1/57 |
| Agent outcomes | 38 first-try passes, 21 repaired passes, 10 refused for workflow, 2 repair failures, 25 stopped after a sibling failed | 39 first-try passes, 53 repaired passes, 1 refused for workflow, 3 stopped |
| Q5 verdicts | 14 CalibrationPass, 10 ClientFailed | 23 CalibrationPass, 1 ClientFailed |

- **hf2q failure signature.** The repair-turn reasoning diagnoses E0282 correctly, for example "The compiler
  needs a type annotation for the closure parameter. Let me fix that." The emitted `write_file` is still
  byte-identical to the failing first write. On the next turn the agent writes again instead of calling
  `run_tests`, and the workflow refuses it. Of the 16 identical rewrites:
  - 14 call `write_file` again on turn 5 (8 of those rewrites change the form, 2 do not, 4 were cut mid-stream
    by the coordinator);
  - 2 call `run_tests`.
  
  The turn-4 reasoning never equals the turn-2 reasoning (0/16), so a turn is not being replayed whole.
- **Engine difference.** Identical repair rewrites are 16/43 on hf2q vs 1/57 on the peer, Fisher two-sided
  p = 2e-6. This confirms the pre-registered concern. The repair workload lifts both engines (the floor is
  18/24), but hf2q stays BelowFloor while the peer passes 23/24.
- **Next step (not started; the operator holds the GPU decision).** Run a same-prefix fork replay of the
  repair turn on both engines, with the recorded turn-4 requests and greedy decoding, and a logprob trace
  of the `write_file` argument tokens. Suspects to separate, as hypotheses only:
  - hf2q's Qwen3.6 decode route on the repair turn: `auto` speculation (MTP/history proposals) or joint
    rows;
  - the repetition-penalty history scope (generated-only in hf2q vs the peer's prompt-inclusive 64-token
    window);
  - numerics.

Analysis: `analyze_repair.py` writes `repair-analysis.json` in the evidence directory.

### Repair-turn fork replay (2026-11-04): why hf2q re-emits the failing file

**Setup.** Evidence is in `$V/q4-qwen36/fork2/`. The replayed requests are the recorded R1 turn-4 (repair)
requests, sent byte-for-byte, so both engines see the same prefix. The cases are:
- the 16 hf2q identical-rewrite agents;
- 16 hf2q agents whose repair changed the file, 8 of them from the same instances.

Decoding is greedy and unary on one memguarded server per session. hf2q is the v0.1.29 release binary
`86b64b02…`, the peer is e86f5fd1 `4b604952…`, and the GGUF is `815a22ca…`. Modes:
- **single:** first send, then the same request at once (cold: 0 cached tokens; then warm: about 1,830
  cached);
- **logprobs:** `top_logprobs` 20;
- **chain:** turns 0–4 replayed to rebuild the agent's own state;
- **conc:** every agent of an instance replayed together.

**Identical rewrites of the 16 cases on the same prefix** (the controls stay at 1–4/16 on hf2q and 0–1/16 on
the peer in every arm):

| Arm | single (cold / warm) | logprobs | chain | conc |
|---|---:|---:|---:|---:|
| hf2q default (rep 1.05, speculation `auto`, TQ K/V) | 12 / 12 | 14 | 15 | 16 |
| hf2q `HF2Q_QWEN_SPECULATION=off` | 14 / 14 | 14 | 14 | — |
| hf2q dense K/V (`HF2Q_TQ_KV=0`) | 14 / 14 | 14 | — | — |
| hf2q repetition penalty 1.0 | 7 / 7 | 7 | 6 | — |
| hf2q penalty 1.0 and speculation off | 7 / 7 | 7 | 7 | — |
| hf2q penalty 1.0 and dense K/V | 7 / 7 | 6 | — | — |
| Peer default (rep 1.05, 64-token window with prompt) | 0 / 0 | 0 | 0 | 1 |
| Peer, repetition penalty 1.0 | 1 / 1 | 1 | 0 | — |

**Verdict per suspect:**
- **Prefix/KV reuse: excluded.** Cold and warm give identical results; rebuilt chains and concurrent
  replays reproduce the rewrite, and turns 0–3 of the chains reproduce the recorded hf2q turns 124/128.
  Dense K/V gives the same rate, and on equal prefixes its fork preferences equal TQ K/V's (median
  difference +0.01 nat, 43 points).
- **Speculation or joint route: excluded.** Speculation off gives 14/16, and the speculation-ineligible
  logprobs route gives 14/16. Joint decode is off for this MoE worker (`joint_decode=false`), and
  single sequential requests reproduce the rewrite.
- **Tool grammar: a separate format defect, not the cause.**
  - hf2q's auto-lazy Qwen tool grammar starts the body at `<function=` right after the `<tool_call>`
    trigger. It therefore suppresses the template's newline: hf2q emits `<tool_call><function=…` in 192/192
    logprobs outputs, the peer `<tool_call>\n<function=…` in 64/64. hf2q's top-20 after `<tool_call>` holds
    only `<` and `<f`, and the peer's top-1 there is `\n` at about 0 nat.
  - Forcing the peer with hf2q's reasoning and hf2q's newline-less opening (raw `/completion`, vendor
    render, token counts equal to the recorded 1,721…, `repeat_penalty` 1.0) gives 0/16 rewrites. The same
    forcing with the newline also gives 0/16.
- **Repetition penalty: an amplifier.**
  - At 1.0, hf2q drops from 12–14/16 to 7/16. The peer stays at 0–1/16 with or without its penalty.
  - In 6 of the 14 copy/fix forks, hf2q's raw (pre-penalty) top-1 is the fix, and the 1.05 penalty flips
    the greedy pick to the copy (`probe-summary.txt`).
- **Raw logits (numerics or compute): the root.**
  - The peer was given hf2q's exact generated text up to each copy/fix fork: `s.parse` followed by `().` or
    `::<`, and `map_err(|e` followed by `|` or `:`. That covers all 32 cases, 63 points.
  - hf2q's raw logits favor the failing form over the fix by a median +1.73 nat more than the peer's
    (+1.81 on identical cases, +1.36 on controls). The shift is positive at 54/63 points (`probe-all-summary.txt`).
  - It is unchanged with dense K/V and with speculation off. The bias is in hf2q's forward pass on the Q4_K
    artifact, not in state reuse, routing, grammar or sampling. The canonical diagnosis measured the same
    direction on the first write (closure-parameter margin, hf2q −0.09 vs peer +0.28 nat). On the
    longer repair-turn prompt the shift is larger.

### Bias localization (2026-11-04): partial-rotary RoPE pairs the wrong dimensions

Evidence: `~/.local/state/hf2q/v0130/q4-qwen36/localize/` (`compare.json`, `lens.json`,
`layer3/stage-000.json`, `layer3/stage-001.json`, `summary.txt`). Same 63 copy/fix points and same token ids
on every engine: hf2q's exact generated ids up to each fork (prompt tokens equal the recorded ones 63/63).

- **Reference runs.** Three engines score the same prefixes:
  - hf2q v0.1.29 `generate`, raw passthrough template, one prefill, speculation off. It agrees with the
    serve route to a median 0.19 nat.
  - The pinned peer (`e86f5fd1`) through a small eval-callback dump tool on the same GGUF.
  - The BF16 safetensors through mlx_lm, read-only (`/opt/hf2q/models/sources/Qwen3.6-35B-A3B-stock`).
- **BF16 sides with the peer.** Copy-over-fix preference relative to the peer:
  - hf2q +1.55 nat median, positive at 52/63.
  - BF16 −0.58, positive at 14/63.
  - hf2q − BF16 +1.95, positive at 59/63.
- **Noise baseline.**
  - Peer f16 vs q8_0 KV: 0.0012 median rel-L2 on the last-token residual at layer 3, rising to 0.054 at
    layer 39. The preference moves by −0.04 nat.
  - Peer ubatch 512 vs 2048: bit-identical on the 44 points where the 2048 run finished. The dump tool died
    on SIGBUS in the other 19.
  - hf2q KV precision can't serve as a baseline here: the last prefill token attends to fresh K/V. TQ and
    dense KV gave bit-identical logits on point 000.
- **First divergent layer: 3, the first full-attention layer.**
  - Median rel-L2 of hf2q vs peer, last-token residual: 0.0006–0.0008 after the three DeltaNet layers 0–2,
    then 0.082 after layer 3. That is 68× the KV-precision noise at the same layer.
  - Every full-attention layer (3, 7, …, 39) adds +0.03 to +0.10, 0.52 in total. The 30 DeltaNet/MoE layers
    add −0.20 in total (each between −0.033 and +0.014).
- **Stage within layer 3, all 1,784 positions, points 000 and 001.** Median rel-L2 hf2q vs peer:

  | stage | median rel-L2 |
  |---|---|
  | attention input norm | 0.0007 |
  | Q after q-norm | 0.0011 |
  | K after k-norm | 0.0010 |
  | V | 0.0013 |
  | output gate | 0.0003 |
  | Q after RoPE | **0.60** |
  | K after RoPE | **0.54** |
  | attention output, pre-gate | 0.24 |
  | attention block output | 0.27 |

  - At position 0 the two engines agree (0.001), as they must, since RoPE is the identity there.
  - Both engines use the right frequencies, θ_i = 10⁷^(−2i/64).
  - They pair different dimensions. The peer rotates `(x[i], x[i+32])` for i < 32 (NeoX within the 64-dim
    rotary slice). hf2q rotates `(x[i], x[i+128])`.
  - A CPU rotation of hf2q's own q-norm output with pairing `(i, i+32)` reproduces the peer's Q to 0.0011.
    With pairing `(i, i+128)` it reproduces hf2q's to 1e-5.
- **Cause.**
  - mlx-native 0.15.16 `src/shaders/rope_multi.metal` documents the pairing as `(x[p], x[p + head_dim/2])`
    and implements it the same way: `half_dim = head_dim / 2` at lines 178 and 253, and
    `x1 = input[base + pair_idx + half_dim]` at line 226.
  - Only `rope_dim/2` pairs rotate. With partial rotary (Qwen3.5/3.6: `rope_dim` 64 of `head_dim` 256), the
    partner element must be `p + rope_dim/2`, as in the BF16 reference and the peer. The kernel uses
    `p + head_dim/2`.
  - hf2q's Qwen3.5-family full attention calls it through `apply_imrope`/`apply_imrope_into`
    (`gpu_full_attn.rs`) on prefill. Per the code, the same call serves decode, MTP and joint decode
    (`joint_decode_core.rs`, `mtp.rs`) [INFERENCE: decode not dumped].
  - The fused `fused_head_norm_rope_*` kernels pair `(i, i + half_rope)` correctly.
- **How the bias reaches the token.** A logit lens projects each layer's residual onto the copy−fix row
  difference of the BF16 head; the lens reproduces the peer's final preference to 0.025 nat.
  - The wrong relative-position encoding perturbs every full-attention layer.
  - The copy preference itself appears late: hf2q − peer is +0.25 at layer 31, +1.18 at 33 and +1.37 at 35,
    against q8_0-KV noise of 0.07–0.17. BF16 − peer stays at −0.0 to −0.55.

This defect is in hf2q's compute path, not the artifact, and it applies to every Qwen3.5-family model whose
`rope_dim` is smaller than `head_dim` [INFERENCE: only Qwen3.6-35B-A3B measured]. The fix (pair
`p + rope_dim/2`) belongs in mlx-native. R1 and the Q3 Qwen3.6 rows need re-measurement after it. The
grammar newline is still worth a separate fix.

**Fix verified (mlx-native 0.15.18, hf2q main `38eea839`).** The same 63 points and layer-3 dumps were
re-run with a release build of `38eea839`. Peer and BF16 data are reused; evidence is in `summary.txt`,
`compare-ropefix.json`, `lens-ropefix.json` and `layer3/stage-ropefix-*.json`.
- **Layer 3.** Q and K after RoPE differ from the peer by 0.0011 and 0.0010 (were 0.60 and 0.54), the same
  as before RoPE. The attention block output differs by 0.0038 (was 0.27).
- **Every layer.** The last-token residual is within 1.0–1.6× of the peer's KV-precision noise: 0.0016 at
  layer 3 and 0.056 at layer 39, against noise of 0.0012 and 0.054. Before the fix it was 0.082 and 0.32.
- **Copy-over-fix preference relative to the peer.**
  - Median +0.01 nat, positive at 32/63 (was +1.55, positive at 52/63).
  - Median absolute difference 0.16 nat, against peer q8_0-vs-f16 noise of 0.17.
  - The copy form is preferred at 26/63 points (peer 28/63, before the fix 43/63).

**R1 re-run on main (2026-11-05): hf2q 21/24 vs peer 20/24, verdict `Pass`.**
- **Setup.** The frozen R1 design (24 instances, same instance table, case files, GGUF `815a22ca…`, peer arm,
  sampling). The one change is the hf2q binary: origin/main `4bfb0436`, which includes the RoPE fix (mlx-native
  0.15.18), the tool-call newline fix (#686) and #704, merged with this branch's harness as `d7efa085`.
- **Toolchain.** rustc 1.99.1, sha256 `946db5e7…`, cargo 1.99.1 sandbox proof `4b2bea77…`.
- **Plan.** Plan sha256 `e0936e91…`. It differs from the first R1 plan only in the binary, the proof and two
  harness source hashes. Campaign directory: `campaigns/quality-qwen36-repair1-ours-20261105T183923Z`, wall
  time 42 min.
- **Verdict.** `subject_only_fail` 2, `reference_only_fail` 3, `one_sided_p` 0.81. The first R1 was hf2q
  14/24 vs peer 23/24, `Loss`.
- **Against the canonical workload.** On the same instance ids, hf2q passes 16 instances it failed under the
  canonical workload and fails none it passed (two-sided sign test p = 3e-5).
- **The remaining failures match across engines.**
  - Failing-form first writes: hf2q 55/96, peer 58/96.
  - Repairs used and passed: hf2q 51/54, peer 53/57. No repair compiled and failed its test.
  - Failing-form repair writes: hf2q 2/54, peer 3/57.
  - Instances lost: hf2q 3 (i08, i14, i16), peer 4 (i01, i03, i16, i19). In each, one agent ends
    "initial six-step workflow not followed" right after its repair write; in 5 of these 7 the repair
    re-emits the failing first file byte for byte.

### ADR-063 amendment (landed 2026-11-05)

The race workload is recorded in ADR-063 under Q4-QWEN36 Acceptance ("Qwen3.6 race workload"), with an
"Updated" header line and the C2 quality-rule clause naming it as Qwen3.6's declared workload. It records the
operator decision, the rationale and scope, the defect found and fixed on this workload (mlx-native 0.15.18
`rope_multi`, #686), and the result on fixed main (hf2q 21/24 vs peer 20/24, `Pass`).
