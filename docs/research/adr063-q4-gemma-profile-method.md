# ADR-063 Q4-GEMMA (#660): Gemma 4 peer configuration search and per-stage profile — method

Date: 2026-11-03. Host: M5 Max 128 GB. Status: method and tooling. No model was loaded and no GPU run was
made for this document. The only numbers below are re-readings of existing Q3 v9 receipts (§5). The design
is approved for track (a) only: the peer configuration search and the per-stage profile. Track (b),
integrating the Q4-GEMMA-KERNEL primitive, belongs to Q4-GEMMA-ATTN under its own design. The profile makes
no speed claim. Its job is to name the stages Q4-GEMMA-ATTN and any later Gemma change must close.

`$E` = `/Users/robert/.local/state/hf2q/campaigns/adr063-preflight-20261023`; `$G` =
`/Users/robert/.local/state/hf2q/v0130/q4-gemma`. The runner scripts (`$G/*.sh`) are evidence-local, as in
D2, because they bind local paths and hashes. Everything they compute comes from the checked-in tools in §6.

## 1. Fixed inputs

| Input | Identity |
|---|---|
| Target | `gemma-4-26B-A4B-it-hf2q-q4_k_m.gguf`, sha256 `bf65b7e1…8ef53`, from `google/gemma-4-26B-A4B-it@4d7ae498` |
| Peer | the peer server, build `b128-e86f5fd` (sha256 `4b604952…a6c`), the Q3 v9 peer binary |
| Base peer arm | `scripts/four_agent/q5_cases/peer-gemma4.json`: 4 × 262144 unified f16 KV, temperature 0, repeat penalty 1.05, thinking off, `--spec-type none` |
| hf2q | a release build of this branch's commit. Defaults are unchanged; the new diagnostic is off unless set |
| Workload | the canonical four-agent coding client (`qwen_n4_live_smoke.py --family gemma4`), each run a supervised Q5 calibration |

## 2. The peer's valid Gemma 4 speculation options at the pinned build

Source: the peer tree at e86f5fd1, `common/speculative.cpp:74-86` (the type table), `docs/speculative.md` and
`conversion/` (drafter converters). Also used: the HF model API on 2026-11-03 for drafter revisions and LFS
hashes.

| Arm | `--spec-type` | Drafter | Evidence it can run on this target |
|---|---|---|---|
| P0 | `none` | — | Q3 v9 peer arm |
| NM | `ngram-mod` (defaults n-match 24, n-min 48, n-max 64; what `--spec-default` enables) | none | model-free; one hash pool shared by all slots |
| NS | `ngram-simple` (n 12, m 48) | none | model-free |
| NK | `ngram-map-k` (n 12, m 48, min hits 1) | none | model-free |
| NV | `ngram-map-k4v` (8/8/2, n-max 64, the peer documentation's repetition setting) | none | model-free, marked experimental by the peer |
| NC | `ngram-cache` (no lookup files) | none | model-free |
| MT | `draft-mtp` (n-max 3, greedy, backend sampling: D2's P1 flags) | `google/gemma-4-26B-A4B-it-assistant@6e5aaaf4` (839 MB BF16, `Gemma4AssistantForCausalLM`) | the peer's Gemma 4 assistant architecture; its MTP path shares the target KV for Gemma 4 (`speculative.cpp:1504-1581` and the context setup for that architecture) |
| E3 | `draft-eagle3` (n-max 3, the speculator's `speculative_tokens`) | `RedHatAI/gemma-4-26B-A4B-it-speculator.eagle3@226b2692` (1.86 GB; aux layers 2/15/27, draft vocabulary 32000) | listed as supported in the peer's `docs/speculative.md:40`; the peer converter registers `Eagle3DraftModel` |
| DF | `draft-dflash` (n-max 15; block 16) | `z-lab/gemma-4-26B-A4B-it-DFlash@77d42027` (859 MB; Qwen3 backbone, target layers 1/6/11/17/22/27) | converter `conversion/qwen.py:801` reuses the Gemma 4 tokenizer and embedding scale |
| DS | `draft-simple` (n-max 8, n-min 0, p-min 0.75, greedy) | `google/gemma-4-E2B-it@4adf1574` (10.2 GB BF16 safetensors with vision/audio towers; converted text-only at Q8_0), operator decision 2026-11-03 | same 262144-token Gemma 4 vocabulary; the peer's `gemma4` architecture loads E2B (per-layer inputs, 20 KV-shared layers) and its converter lists E2B; the vocabulary compatibility check (`speculative.cpp:108-160`) runs at load |

E3, DF and DS place the whole draft model on the GPU (`--spec-draft-ngl all`), so as not to rely on `auto`
placement next to `--fit off`. MT shares the target context. DS's draft KV takes the target's 1,048,576-token
pool. E2B has 15 KV-owning layers, 3 of them full attention with one 512-dim KV head, so the draft KV is
about 6 GiB at f16 [INFERENCE from the config]. Q5 decides the memory tier.

Not candidates:
- `draft-dspark`. The pinned build supports only Qwen3 DSpark backbones (`docs/speculative.md:106`), and no
  26B-A4B DSpark checkpoint is published (the HF API returns no such repo).

Drafters are reference-only benchmark inputs, never shipped or served by hf2q (operator decisions of
2026-10-31 and 2026-11-03, as in D2). `$G/00-acquire-drafters.sh` downloads each at its pinned revision,
checks the safetensors sha256, and converts it with the peer's own `convert_hf_to_gguf.py`. MT, E3 and DF
convert to BF16; DS converts to Q8_0, text only. EAGLE-3 and DFlash convert with `--target-model-dir` set to
the target's source snapshot. Every heavy step waits while HOST-LOCK exists and runs at background QoS.
Per-conversion receipts are copied to `$G/<arm>.receipt.json`. Combined types (for example
`ngram-mod,draft-mtp`) enter only through §3's extension rule.

### Storage of the drafters (operator, 2026-11-03)

Sources and GGUFs live on the external drive, under `…/hf2q/ext/models/q4-gemma/`. The target GGUF and
both engines stay on the internal disk. The operator allows a drafter on the external drive in timed runs
only if loading from it does not affect makespan. The argument and the check:

- **Argument.** The peer arm runs with `--load-mode none`, which means no mmap. The draft model is loaded
  with the same model parameters (`common_speculative_init_result`, `common.cpp:1708`). Every drafter byte is
  therefore read into allocated buffers during load, before readiness, and Q5's makespan starts after
  readiness. After load, the weights are anonymous memory and are never paged back from the file. Load time
  from the external drive is longer, but it lies outside the makespan.
- **Check.** Each run with an external drafter records the external drive's cumulative `iostat -I`
  transfers every second (`$G/ext_io.py`), and the transfer inside the run's makespan bracket is compared
  with 64 MB. Reads and writes are counted together, so the figure is an upper bound.
- **Rule.** A qualifying arm that fails or cannot bracket the check is marked
  `needs-internal-drafter`. It is not timed until its drafter moves to the internal disk, which is the
  operator's call given the free space.
- **Pre-run guard.** Every run also checks that the internal disk has at least 20 GB free.

## 3. Selecting the best valid configuration (the D2 rule)

1. **Qualification** (`$G/20-peer-search.sh`, first half). One N4 run per arm, with P0 first as the
   transcript reference. An arm qualifies when all of these hold:
   - Q5 passes;
   - 4/4 tasks complete;
   - its replies carry drafted tokens (per-reply `timings.draft_n` > 0).

   Transcripts are compared with P0 by `q1_decision_policy.py transcripts`. Greedy verification should give
   `ExactTranscripts`. An arm whose transcripts diverge but whose tasks pass stays valid and is flagged.
   If P0 fails, the search halts. Any other failing arm is dropped as arm-negative; D2's peer DFlash2 fails
   here, for comparison.
2. **Primary.** Three passes in forward/reverse/forward order over P0 and the qualified arms. A P0 failure
   halts; any other failure drops the arm.
3. **Rank** (D2 `extract.py table`, lexicographic):
   1. four-slot completion;
   2. final tests;
   3. Q5 pass on every run (the memory tier: drafters add weights and draft KV);
   4. median makespan.

   Acceptance rates are reported but never ranked.
4. **Tie-break** (pre-registered). If the top two medians differ by less than 3%, `$G/21-tiebreak.sh` adds
   4 ABBA runs of each and the arms are re-ranked.
5. **Extension** (pre-registered). If the best model-free arm and the best drafter arm both rank above P0,
   their combination (`--spec-type <ngram>,<draft>`) gets one qualification run and 3 runs against the
   leader.

The best valid configuration is the top-ranked arm. Each arm's makespan against P0 carries a 95% bootstrap
CI (`gemma_stage_profile.py compare`). The Q4 campaigns then use that arm, so the gap may widen beyond Q3
v9's 13.4%.

## 4. Per-stage profile method (hf2q and the peer, same workload, host and session)

`$G/40-profile.sh BEST` runs the following on one host session, under one lane grant:

| Layer | Runs | Measures | Distortion statement |
|---|---|---|---|
| L1 request level | 7 ABBA blocks of (hf2q H, peer BEST), unprofiled | per turn: span, prefill, decode, residual, hf2q queue, client/tool, tail; semantic TTFT; SSE events and gaps | none added: both engines already report per-request timing (hf2q `x_hf2q_timing`, the peer `timings`), and the client already records monotonic SSE events |
| L2 GPU timeline | 3 blocks of (H, BEST) under `gemma_mst_capture.py` | aligned to request spans: GPU busy (union), GPU idle while a request is in flight (host dispatch/encode/scheduling/sampling/delivery), no-request time, GPU busy in prefill windows vs decode-only | traced vs L1 makespan per engine (`distortion-*.json`) |
| L3 hf2q GPU classes | 3 blocks of (HC, H): `HF2Q_DECODE_CATSPLIT=1 HF2Q_GEMMA_PREFILL_CATSPLIT=1`; 2 of (HH, H): `HF2Q_HOST_PHASES=1` | decode body: attention, MoE, dense, norm; warm prefill: pre-attention, attention, dense, MoE; host phases | HC/H and HH/H makespan ratio, paired by block. The splits serialize command buffers, so they give shares of GPU-busy time, never step times |
| CF warm attention | 3 blocks of (HT, H): `GEMMA_WARM_TILED_ATTENTION=1` vs default vec-hybrid | warm-turn prefill and makespan delta of the attention route | unprofiled; HT is the kill-switch counterfactual, not a quality-valid default |

### Mapping to the acceptance's stages

- **Scheduling.** Comes from L1:
  - residual (span − prefill − decode) and hf2q queue per turn, especially cold turn 0, where four requests
    arrive together;
  - decode time spent sharing steps with other slots' prefill quanta;
  - from L2, GPU-idle-in-flight.
- **Dispatch/encode.** The L2 GPU-idle-in-flight time minus the residual that the client clock attributes
  to queue and delivery. For hf2q, the L3 host phases split this further (body/head waits, readbacks,
  sample loop, scheduler step).
- **Attention and MoE.** Shares from the L3 catsplits, applied to the L1 unprofiled prefill and decode times
  for hf2q.
  - The peer exposes no per-kernel GPU timer at e86f5fd1. Its debug groups exist only in Xcode capture mode
    (`ggml-metal-ops.cpp:546-559`), and Metal System Trace from the CLI cannot enable the shader timeline
    (see `scripts/aggregate-q4_0-mst.py`; xctrace 27.2 `--show-recording-options` offers no such option).
  - The peer's attention-vs-rest split is therefore model-based: the L1 warm-prefill regression
    `prefill ~ b0 + b1·new + b2·new·(cached + new/2)`, whose b2 term scales with attention work. It is
    stated as a fit with its R², not a measurement. A peer kernel class that cannot be attributed is
    recorded as Unknown.
- **SSE cadence.** From L1:
  - per-slot worst semantic gap;
  - semantic events per streamed turn;
  - the share of excess gap time (gaps > 2× the run median) that overlaps another agent's prefill window
    (co-scheduled prefill) versus the slot's own emission cadence.
- **Share attributable to warm attention.** CF gives the route's makespan and prefill delta. L3 gives
  attention's share of warm-prefill GPU time. Together they bound what Q4-GEMMA-KERNEL/ATTN can recover.
- **Share of the stock SSE-gap regression.** L1 SSE analysis of stock vs current hf2q gives the worst-gap
  turn, its event position and its relation to decode step time. It can be re-read on the Q3 v9 stock and
  experiment receipts without new runs.

Gap attribution means: per stage, the median paired-by-block difference (hf2q − peer) with a 95% bootstrap
CI (10,000 resamples, seed 6306), and its share of the makespan difference. Stage shares need not sum to
exactly 1, because each is a median of differences.

## 5. What the existing Q3 v9 receipts already show (L1 only; peer P0, not the best configuration)

Command (read-only):

```sh
python3 -B scripts/four_agent/gemma_stage_profile.py compare \
  $E/q3-campaign-v9/gemma4-all/runs --arm 'experiment=n4-*-experiment-*' --base 'peer=n4-*-peer-*' \
  --json $G/layer1-v9/experiment-vs-peer.json
```

Output sha256: `5f858dee…583f`. The run set is the 21 complete blocks; hf2q `5d421a2c` (tiled warm
attention off); peer P0. Values are medians of the per-run mean over agents, in seconds, paired by block:

| Stage | hf2q | peer | Δ (95% CI) | Share of Δ makespan |
|---|---:|---:|---:|---:|
| makespan | 15.109 | 12.844 | +2.037 (+1.395, +2.859) | — |
| prefill | 3.579 | 2.600 | +0.894 (+0.480, +1.027) | 0.44 |
| decode | 9.259 | 8.137 | +1.164 (+0.833, +1.255) | 0.57 |
| residual | 0.827 | 0.780 | +0.064 (−0.028, +0.337) | 0.03 |
| client/tools | 1.147 | 1.087 | +0.027 (−0.009, +0.190) | 0.01 |

Per turn (medians over agents and runs, in ms; new and cached are hf2q token counts):

| Turn | span h/p | prefill h/p | decode h/p | residual h/p | new / cached | SSE events h/p |
|---|---|---|---|---|---|---|
| 0 cold | 1538 / 982 | 276 / 524 | 755 / 353 | 508 / 98 | 465 / 0 | unary |
| 1 | 636 / 660 | 159 / 141 | 430 / 471 | 43 / 53 | 58 / 458 | 3 / 11 |
| 2 | 3568 / 3265 | 456 / 307 | 3089 / 2886 | 24 / 68 | 191 / 516 | unary |
| 3 | 962 / 840 | 553 / 357 | 389 / 350 | 21 / 103 | 220 / 707 | 3 / 7 |
| 4 | 1285 / 1009 | 835 / 545 | 400 / 346 | 23 / 128 | 301 / 927 | unary |
| 5 | 751 / 672 | 421 / 255 | 307 / 260 | 21 / 160 | 132 / 1246 | 3 / 7 |
| 6 | 4548 / 3993 | 1059 / 563 | 3577 / 3268 | 28 / 178 | 361 / 1392 | unary |

Leads these numbers give the profile to confirm or refute. None is a measured stage attribution yet.

- **Cold turn 0 (scheduling).** hf2q's own prefill is faster (276 vs 524 ms), but the four simultaneous
  cold requests wait for one another. hf2q has a 423 ms server queue (`x_hf2q_timing` total − prefill −
  decode) and decode inflated by other slots' prefill quanta. The turn-0 span is 556 ms longer.
- **Warm prefill (turns 3–6).** 1.5–1.9× the peer per turn. The hf2q regression fit is poor (R² 0.56, and
  its attention term comes out negative). That is expected: a joint warm cohort's whole forward is charged
  to every lane, so request-level timing cannot separate attention for hf2q. That is why the warm-prefill
  split (§6) exists. The peer's fit is R² 0.83, with the attention term about 10% of its warm prefill.
- **Long decodes (turns 2 and 6, about 150 tokens).** 7–9% slower per token.
- **SSE cadence.** hf2q emits each streamed tool call in 3 semantic events; the peer emits 7–11. hf2q's
  first semantic event arrives later: turn 1 at 573 vs 380 ms, turn 3 at 940 vs 685 ms. hf2q's worst-slot
  gap is about 60 ms, always on turn 1, between its last two events. It is the same in stock (59 ms) and
  experiment (62 ms), although their decode steps differ (36.6 vs 22.9 ms). So the stock SSE-gap
  regression is not decode cadence; L1 must locate it in end-of-call emission. hf2q's excess gaps never
  overlap another agent's prefill window (the excess-overlap share is under 0.1% in all 21 runs). The
  peer's share is 0 in 20 of its 21 runs and 99% in one.

## 6. Code (branch `feat/q4-gemma-profile`, local commits only)

- `src/serve/gemma_prefill_catsplit.rs` and boundaries in `src/serve/forward_prefill_batched.rs`, with a
  worker-exit emit in `src/serve/api/engine.rs`. This is `HF2Q_GEMMA_PREFILL_CATSPLIT=1`. It is off by
  default, and only the exact value `1` enables it.
  - In each layer of a warm live-prefix-resume batched prefill, it closes GPU-timed command buffers
    (`finish_with_gpu_time`) at four points: before and after live attention, before the routed MoE, and
    at the layer end. The four categories are `pre_attn`, `attention`, `dense`, `moe`.
  - It changes no dispatch, and it is skipped under `HF2Q_GRAPH_OPT_PREFILL=1`.
  - The worker prints one `[PREFILL_CATSPLIT_JSON]` line at graceful exit. SIGTERM reaches the worker-exit
    dump through the HTTP drain and `engine.shutdown()`.
  - Not covered: cold prefill, the setup before layer 0, the head, and suffixes under 32 tokens.
  - The general category-split wiring of ADR-065 E1 is not part of this change. This split covers only the
    Gemma warm prefill that E1's decode classes do not.
- `scripts/four_agent/gemma_stage_profile.py`: L1 profile and compare, warm-prefill fit, SSE attribution,
  L2 alignment and classes, and the L3 dump parser. Reads receipts only.
- `scripts/four_agent/gemma_peer_spec_arms.py`: the §2 arm cases, derived from `peer-gemma4.json`.
- `scripts/four_agent/gemma_mst_capture.py`: one Q5 run under an all-process Metal System Trace. It keeps
  only the credential-scanned GPU-schema exports and deletes the raw bundle (the ADR-042 rule).
- Unit tests:
  - `test_gemma_stage_profile.py`, `test_gemma_peer_spec_arms.py` and `test_gemma_mst_capture.py`;
  - in Rust, `gemma_prefill_catsplit::tests`.

## 7. Commands and expected duration

```sh
export RUNNER_COMMIT=<feat/q4-gemma-profile commit>
bash $G/05-build.sh                  # waits for HOST-LOCK; focused test + release build to $G/bin-<commit>/
bash $G/00-acquire-drafters.sh       # CPU/disk/network only, preempted by any HOST-LOCK: ~13.8 GB download,
                                     # 3 BF16 + 1 Q8_0 conversions onto the external drive
bash $G/10-make-cases.sh             # seconds; adds MT/E3/DF/DS once $G/drafters.json exists
bash $G/20-peer-search.sh            # GPU: qualify 10 runs + primary up to 30 runs
bash $G/21-tiebreak.sh A B           # GPU, only if the top two are within 3%: 8 runs
bash $G/40-profile.sh BEST           # GPU: L1 14 + L2 6 (traced) + L3 10 + CF 6 = 36 runs
```

**Expected duration.** Measured from the Q3 v9 receipts, a supervised Gemma N4 run takes 21–31 s from Q5
start to finish, with the model page-cached. That covers load, readiness, a makespan of about 13–15 s and
teardown; the peer arm took 21–24 s. Drafter arms add the drafter hash and load. The first load from the
external drive also adds about 5–15 s at the drive's rate [INFERENCE]. Allowing 30–40 s per untraced run:

| Step | Runs | GPU lane |
|---|---:|---|
| Peer search (qualify + primary) | 40 | 20–30 min |
| Tie-break and extension (if triggered) | ≤ 12 | ≤ 8 min |
| Profile L1 + L3 + CF | 30 | 15–20 min |
| Profile L2, traced | 6 | 10–20 min (xctrace start, stop and export add 1–3 min per run [INFERENCE]) |
| **Total** | ≤ 88 | **about 1–1.5 h**, including halts and inspection between steps |

The earlier estimate of 3.5–4.5 h assumed 2–2.5 min per run and was wrong.

## 8. Open points before the GPU lane

- **Trace clock.** The xctrace `metal-gpu-execution-points` timestamps are assumed to be relative to
  recording start, with the anchor taken from the first "Starting recording" line. The alignment search
  (±5 s at 1 ms resolution, minimizing GPU-busy time outside request spans) tolerates start latency but
  not a different time base. The first L2 run must check `busy_outside_spans_s ≈ 0` before the rest are
  trusted.
- **Trace size.** All-process Metal System Trace over a Q5 run, model load included, is expected to be
  hundreds of MB before export. The raw bundle is deleted after export.
- **Drafter memory.** Each separate drafter's KV is sized to the target's 1,048,576-token pool. EAGLE-3 and
  DFlash each have one full-attention layer, about 4 GiB at f16; E2B needs about 6 GiB [INFERENCE]. Q5
  decides the memory tier.
- **External drive.** The drafters are timed from the external drive only while the per-run witness in §2
  passes.
