# ADR-063 Q4-GEMMA (#660): Gemma 4 peer best configuration and per-stage profile — result

Date: 2026-11-04. Host: M5 Max 128 GB, under the GPU lane granted to GemmaProfile, with HOST-LOCK held for
the whole timed sequence. The method is in `adr063-q4-gemma-profile-method.md`. This record makes no speed
claim. It names the stages Q4-GEMMA-ATTN and any later Gemma change must close.

`$G` = `/Users/robert/.local/state/hf2q/v0130/q4-gemma`. Every number here comes from raw receipts under `$G`,
read by the checked-in `scripts/four_agent/gemma_stage_profile.py` (read-only).

## Identities

| Input | Identity |
|---|---|
| Runner / hf2q source | `feat/q4-gemma-profile` at `7f295b38` (rebased on main `ff4af4bd`); release binary `$G/bin-7f295b38/hf2q`, sha256 `f00a0591…074c`, defaults unchanged |
| Peer | build `b128-e86f5fd`, sha256 `4b604952…a6c` |
| Target | `gemma-4-26B-A4B-it-hf2q-q4_k_m.gguf`, sha256 `bf65b7e1…8ef53` |
| Drafters | MT `e1145ee2…`, E3 `88110608…`, DF `99316f42…`, DS `01732465…` and DS2 `d5d772cb…` (§1); loaded from the internal disk, receipts `$G/*.receipt.json` |
| Cases | `$G/cases/{peer,hf2q}/cases-receipt.json` and `$G/cases/peer/G-DS2-draft-e2b-vocab26b.json` |

Q5 verdicts: all 46 search runs passed (11 qualification, 35 primary), and 30 of 31 profile runs passed. The one
failure is the counterfactual HT run (§5).

## 1. The peer's best valid Gemma 4 configuration: MT (`draft-mtp` with the Gemma 4 assistant drafter)

Qualification: one run per arm, with P0 as the transcript reference (`$G/search/qualify/summary.tsv`).
Every arm passed Q5 with 4/4 tasks and `ExactTranscripts` against P0. Two arms drafted nothing:
- **NM (`ngram-mod`).** Zero drafts. Its 48-token minimum draft exceeds most replies.
- **DS (E2B plain draft).** The peer refused speculation at load ("the target and draft vocabs are not
  compatible"). An offline comparison of the two GGUFs' tokenizer metadata found exactly one differing token
  text: id 249788 is the target's `𡉺` but E2B's `<glitch_pruned_249788>`.
  - DS2 is the same Q8_0 E2B with the target's token list (`$G/07-e2b-vocab-align.py`). Tensors and all
    other metadata are unchanged.
  - DS2 qualified (1871 drafted tokens, `ExactTranscripts`) and joined the primary.

Primary: three passes in forward/reverse/forward order. The top two (MT and P0) were within 3%, so the
pre-registered tie-break added 4 more runs each. DS2 ran 3 times interleaved (`$G/22-ds2-tiebreak.sh`).
Ranking: completion, final tests, Q5 memory tier, median makespan (`$G/search/primary/extract.json`):

| Rank | Arm | Runs | Median makespan s | Accepted per drafted | Worst headroom GB |
|---:|---|---:|---:|---:|---:|
| 1 | MT draft-mtp (assistant) | 7 | 13.42 | 0.79 | 45 |
| 2 | P0 none | 7 | 13.77 | – | 52 |
| 3 | NS ngram-simple | 3 | 14.06 | 0.05 | 52 |
| 4 | NK ngram-map-k | 3 | 14.20 | 0.03 | 52 |
| 5 | NV ngram-map-k4v | 3 | 14.40 | 0.36 | 52 |
| 6 | DF draft-dflash | 3 | 14.53 | 0.23 | 39 |
| 7 | NC ngram-cache | 3 | 15.04 | 0.25 | 52 |
| 8 | DS2 draft-simple (E2B, target token list) | 3 | 25.78 | – | – |
| 9 | E3 draft-eagle3 | 3 | 27.10 | 0.19 | 29 |

Every arm has completion 1.0, quality 1.0 and 3/3 (or 7/7) Q5 passes. MT is −0.35 s against P0, with a 95%
bootstrap CI of (−1.21, +0.08) s (`final-MT-vs-P0.json`). It ranks first under the rule but is not separated
from P0 at this sample size. The extension rule (an n-gram + drafter combination) is not triggered, because
no n-gram arm ranks above P0. The Q4 Gemma campaigns use MT.

## 2. Per-stage gap, hf2q vs the peer's MT (L1, 7 paired blocks, unprofiled)

`$G/profile/l1/compare-hf2q-vs-peer.json`. Values are medians of the per-run mean over agents, paired by block:

| Stage | hf2q s | peer MT s | Δ s (95% CI) | Share of Δ makespan |
|---|---:|---:|---:|---:|
| makespan | 14.867 | 12.477 | +2.734 (+2.285, +3.072) | — |
| prefill | 3.862 | 2.763 | +1.056 (+1.005, +1.266) | 0.39 |
| decode | 9.203 | 7.958 | +1.170 (+0.961, +2.012) | 0.43 |
| queue/admission/delivery (residual) | 0.669 | 0.687 | −0.035 (−0.068, +0.036) | −0.01 |
| client/tools | 1.113 | 1.014 | +0.100 (+0.054, +0.128) | 0.04 |
| tail | 0.297 | 0.041 | +0.257 (−0.009, +0.676) | 0.09 |

The peer in its best configuration finishes 16.1% sooner than hf2q by the ratio of medians (hf2q/peer = 1.19;
the median paired-block difference is 2.73 s). The gap widened
from Q3 v9's 13.4%, as the acceptance allowed.

Per turn (medians over agents and runs, in ms; queue is hf2q's server-side wait):

| Turn | span h/p | prefill h/p | decode h/p | residual h/p | hf2q queue | new / cached | SSE events h/p | semantic TTFT h/p |
|---|---|---|---|---|---:|---|---|---|
| 0 cold | 1563 / 1067 | 275 / 574 | 772 / 313 | 515 / 174 | 422 | 465 / 0 | unary | – |
| 1 | 668 / 632 | 163 / 152 | 460 / 440 | 44 / 47 | 0 | 58 / 458 | 3 / 11 | 603 / 372 |
| 2 | 3649 / 3192 | 468 / 320 | 3157 / 2820 | 22 / 78 | 19 | 191 / 516 | unary | – |
| 3 | 995 / 791 | 573 / 372 | 401 / 341 | 20 / 81 | 0 | 220 / 707 | 3 / 7 | 973 / 650 |
| 4 | 1267 / 934 | 844 / 532 | 402 / 313 | 21 / 90 | 17 | 295 / 927 | unary | – |
| 5 | 781 / 706 | 445 / 256 | 318 / 292 | 20 / 152 | 0 | 132 / 1240 | 3 / 7 | 759 / 530 |
| 6 | 4552 / 4072 | 1085 / 546 | 3438 / 3399 | 21 / 133 | 17 | 355 / 1386 | unary | – |

- **Scheduling (cold turn 0).** The four simultaneous cold requests cost hf2q a 422 ms server queue plus
  459 ms more decode, which here means time sharing steps with other slots' prefill quanta. hf2q's own
  prefill is faster (275 vs 574 ms). Net: the turn-0 span is +496 ms.
- **Warm prefill (turns 1–6).** +11, +148, +201, +312, +189 and +539 ms per turn, about +1.4 s per agent.
  The warm-prefill fit (168 points per engine) gives:
  - hf2q: 2.13 s per 1k new rows and 0.346 s per 1M row×context (R² 0.84);
  - peer: 1.18 s and 0.151 s (R² 0.87).

  hf2q's per-row cost is 1.8× the peer's and its context-proportional cost 2.3×. The fit is model-based;
  §4 has the measured split.
- **Long decodes (turns 2 and 6).** +337 and +39 ms. The peer's MT drafts at 0.79 accepted per drafted.
- **SSE cadence.**
  - hf2q's worst-slot gap is 62–66 ms in all 7 runs, always on turn 1, and coincides with another agent's
    prefill window less than 0.1% of the time.
  - The peer's worst gap is 98–190 ms, on turn 1 in 6 of 7 runs, from draft-verify bursts; 2–27% of its
    excess gap time overlaps other agents' prefill windows.
  - hf2q's semantic first-token time is later on every streamed turn (603/973/759 vs 372/650/530 ms). It
    emits each tool call in 3 events against the peer's 7–11.
  - Against MT, the SSE-gap metric favours hf2q (0.064 vs 0.125 s). The stock SSE-gap regression (hf2q vs
    its own stock binary) was not re-run here. The Q3 v9 re-reading in the method doc §5 stands: about
    60 ms, at the end of the turn-1 tool call, and not decode cadence.

## 3. GPU timeline (L2, Metal System Trace, 3 blocks)

`$G/profile/l2/*.gpu-classes.json`. All-process traces; only credential-scanned GPU tables are retained and
the raw bundles were deleted.
- Alignment: the offset search landed 89–179 ms after the recording-start anchor.
- GPU busy outside request spans (within ±5 s): 138–139 ms per run for hf2q and 39–40 ms for the peer. This
  confirms the trace time base. hf2q's constant 0.14 s of GPU work outside spans is not yet attributed.

Per traced run, in seconds, over the request window [first send, last terminal]:

| | window | GPU busy | GPU idle, request in flight | no request | busy in prefill windows | busy, decode only |
|---|---:|---:|---:|---:|---:|---:|
| hf2q (3 runs) | 15.38–16.01 | 12.13–12.85 | 2.09–2.11 | 1.07–1.15 | 4.44–4.62 | 7.67–8.24 |
| peer MT (3 runs) | 12.01–12.39 | 8.50–8.77 | 2.58–2.67 | 0.82–1.05 | 3.06–3.10 | 5.40–5.71 |

**The gap is GPU work, not host dispatch or scheduling idle.** hf2q is GPU-busy 3.4–4.1 s longer per run,
while its idle-with-a-request-in-flight (dispatch, encode, sampling, delivery) is about 0.5 s *smaller* than
the peer's. Splitting the extra GPU time: about 1.4–1.5 s falls in prefill windows and 2.0–2.5 s in
decode-only time.

Distortion, traced vs the L1 untraced medians: hf2q +0.52 s (+3.5%, CI +0.10 to +1.67), peer +0.26 s (CI
−0.57 to +0.35; `distortion-*-traced-vs-l1.json`).

## 4. hf2q GPU classes (L3, profiled; shares of serialized GPU-busy time)

Warm prefill, from the new `HF2Q_GEMMA_PREFILL_CATSPLIT` (3 runs; 6–7 warm forwards per run, 24 lanes,
about 4,990 rows): attention is 1.53 s per run. Shares are the mean of the 3 runs:

| Category | GPU s per run | Share |
|---|---:|---:|
| live warm attention (vec-hybrid) | 1.53–1.54 | 48% |
| routed MoE + layer tail | 1.09–1.11 | 34% |
| dense (O-proj, dense FFN, router, MoE input norm) | 0.35 | 11% |
| pre-attention (norm, QKV, head norm, RoPE, KV write) | 0.24 | 7.5% |

Decode body (`HF2Q_DECODE_CATSPLIT`, one command buffer per category; lm-head and sampling excluded):
- routed MoE 40% (gate/up 21.8%, down 15.3%, other 3.1%);
- dense projections and dense FFN 37.5%;
- attention 17.8%;
- norms and embedding 4.5%.

Host phases (`HF2Q_HOST_PHASES`, 2 runs): the sample loop (argmax, detokenize, scheduler advance) takes
0.64–0.66 s of 8.7–9.1 s of `decode_batch` time per run (7%); the logits readback takes 34 ms.

Distortion:
- **HC (both splits) vs H.** Makespan 59.8 vs 14.2 s (+45.3 s), almost all of it decode (+41.1 s). The
  decode split serializes about 14,000 command buffers per run, so it is a relative ranking only. Warm
  prefill under HC is +0.47 s (+13%), which bounds the prefill split's own perturbation. The decode split's
  4.2× run length confounds it.
- **HH vs H.** +0.48 s (+3.4%, CI −0.07 to +1.20).

## 5. Warm-attention counterfactual (CF): unavailable

The first tiled-F16 warm-attention run (HT, `HF2Q_GEMMA_WARM_TILED_ATTENTION=1`) failed its task: agents 0 and
2 failed with "tool arguments/workflow drift" (Q5 `ClientFailed`, `$G/profile/cf/031-b01-ht-*`). This is
the known #544 decision flip. Per the halt-on-failure rule, the remaining CF runs were not taken. The
warm-attention share therefore rests on §4's measured split (48% of warm-prefill GPU time) and the L2
prefill-window GPU excess (about 1.4–1.5 s per run), not on a makespan counterfactual.

## 6. Stages the Gemma work must close

1. **Warm-prefill attention** (Q4-GEMMA-KERNEL/ATTN): 48% of hf2q's warm-prefill GPU time, about 1.5 s per
   run. The warm-prefill gap is about +1.4 s per agent. This is the largest single attributable class. Even
   removing it fully would not close the 2.73 s gap alone.
2. **Decode GPU work**: +2.0–2.5 s decode-only GPU per run. Routed MoE (40%) and dense projections and FFN
   (38%) dominate. Part of this is the peer's MTP, which removes target decode steps (MT vs P0: −0.48 s of
   decode). This is out of Q4-GEMMA-ATTN's scope; it needs its own design (Gemma MTP, or MoE/dense decode
   kernels).
3. **Warm-prefill per-row cost** (MoE and dense): 1.8× the peer per new row in the fit; 45% of hf2q's
   warm-prefill GPU time.
4. **Cold-turn admission**: about +0.5 s per agent on turn 0 (422 ms queue plus decode inflation). This is a
   scheduling change, not a kernel.
5. **Not causes**: host dispatch/encode gaps (hf2q's in-flight GPU idle is lower than the peer's) and the
   queue/delivery residual after turn 0. SSE cadence affects semantic TTFT and the fairness metric, not
   makespan.

## Evidence

- `$G/search/qualify/`, `$G/search/primary/`, including `extract.{json,md}` and `final-{MT,DS2}-vs-P0.json`.
- `$G/profile/l1/compare-hf2q-vs-peer.json`, `$G/profile/l2/` (traced runs, `*.mst/capture.json`,
  `*.gpu-classes.json`, `distortion-*.json`), `$G/profile/l3/` (`*.catsplit.json`, `compare-*.json`, raw
  `server.stderr`), `$G/profile/cf/`.
- Logs: `$G/search.log`, `$G/ds2-tiebreak.log`, `$G/profile.log`, `$G/05-build-48369211.log`.
