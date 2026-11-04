# ADR-063 Q4-DS4 (#661): the peer on ds4's DeepSeek-V4 file — fit record

Status: measured 2026-11-03 on the M5 Max 128 GB host. Independent review: ACCEPT-WITH-CONDITIONS
(`$V/q4-ds4/review-fit-v1.md`); this record applies conditions C1–C4. `$V` = `/Users/robert/.local/state/hf2q/v0130`.

## Verdict: Fit (fit gate cleared)

The same-GGUF peer, in its matched four-agent settings, completes the largest declared workload (4 slots × 262,144
context, the canonical four-agent coding task) on ds4's DeepSeek-V4 file with Q5 `CalibrationPass`. It did so in
**six of six valid runs**, every agent passing in every run. DeepSeek-V4 is therefore not Unavailable on this file.

This story clears only the fit gate of Q4-DS4-RACE. RACE still needs its other predecessors (ADR-066 N1/N3/N4
IQ2_XXS subset, Q5) and its recorded design approval before it starts.

The file is a benchmark comparison artifact, not a substitute for hf2q conversion.

## Identities

| Item | Value |
|---|---|
| Peer binary | `4b604952a8ac42121974fec1124b34bdeb345383450f0c17c5a20a4d16458c6b`, `/props` build `b128-e86f5fd` (source `e86f5fd1`) |
| Peer artifact | `DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix-0731.gguf`, 86,720,111,488 bytes, sha256 `ca22ae2f838e14077c22bc1c1417b71b45b5e5a3687bd96c2ac6e17fdb6261c0` |
| hf2q binary | v0.1.29 release `86b64b029c6f717f3b6c58654c2aa5b29f420ffa93fe09f31348755fd2df48f2` (tag `v0.1.29` = `c40fbf37`) |
| hf2q artifact | `DeepSeek-V4-Flash-0731-hf2q-deepseek4-agentic-q2.gguf`, 107,431,343,264 bytes, sha256 `50df7e3dfe06321613c78075df4111b14985a0db31087992d7438f2895a3aad6` |
| Peer case | `scripts/four_agent/q5_cases/peer-deepseek4-ds4file.json`, sha256 `72cc4f6e…` (commit `61aa5171`) |
| hf2q cases | `q5_cases/deepseek4.json` `d362f00e…` (H1, trace log filter); derived stage-profile case `06265819…` (H2, `HF2Q_DEEPSEEK_STAGE_PROFILE=1`) |
| hf2q receipts | H1 `33e4aceecbaed9d73ebbe254e0207373c225ae34a1648d745e761ff193e30cb9` (CalibrationPass, 102.96 s); H2 `d27e0268f98b39f46264c379514a689911aa8e51fb656981238874b1348798a6` (CalibrationPass, serial graph, makespan not compared) |
| Chat template (both arms) | reviewed DeepSeek-V4 0731 template `aec3d906…` |
| Sandbox proof / approvals | `owned-cleanup-sandbox-capability-20261031T222313Z/proof.json` `8b286684…`; `approval-templates-proof-8b286684/{peer-deepseek4,deepseek4}-client-approval.json` |

**Artifact sha provenance.** No run receipt re-hashes the peer artifact: Q5 records `sha256_source: "operator-provided,
not verified by the supervisor"`. The binding rests on the operator's local re-hash of 2026-11-02 (#661 body), the
HF LFS sha in the ds4 peer build receipt (`antirez/deepseek-v4-gguf` rev `f71f23d5`), and the unchanged size and mtime
(2026-10-31 20:00, before the re-hash). A supervisor re-hash was not taken because another GPU lane held the host.

**Template.** The ds4 file embeds its own template (sha256 `872492071c22…`, 4,988 chars) with no `reasoning_effort`
tiers; the four-agent client refuses it. The case serves the reviewed hf2q template with `--chat-template-file`
(the override wins in the peer's template loader), so both arms render byte-identical prompts.

## Fit evidence

Peer device memory at exit (run r2, MiB): `110100 total = 19341 free + (90757 self = 81687 model + 7141 context +
1928 compute)`. The KV buffers are allocated for the full 4 × 262,144 up front: CSA 5,376, lightning indexer 1,344,
HCA 160, SWA 215 MiB, plus 46.6 MiB of compressor state. No run logged a Metal out-of-memory or compute error. Every
run: `stop: null`, `cleanup_proven: true`, pressure normal throughout, no blind samples, thermal state 0. Peak host
use 103.1–111.6 GiB (minimum free+cached 16.4–24.9 GiB against a 2.0 GiB effective floor).

| Run | Load mode / ubatch | Q5 verdict | Makespan (s) | Completion tokens | Peak host GiB | receipt sha256 |
|---|---|---|---|---|---|---|
| f1 | auto / 512 | ClientFailed: stale sandbox proof, 0 model requests (neither fit nor no-fit) | — | 0 | 110.0 | `5c8ca69a84c9` |
| f1b | auto / 512 | CalibrationPass | 71.80 | 3,517 | 111.1 | `9730bee6329b` |
| s1 | auto / 1024 | CalibrationPass | 68.80 | 3,529 | 110.9 | `3d14fb9479a0` |
| s2 | auto / 2048 | CalibrationPass | 70.23 | 3,272 | 111.6 | `bd387e03c052` |
| s3 | none / 1024 | CalibrationPass | 65.67 | 3,221 | 104.6 | `91c0dc937784` |
| r1 | auto / 1024 | CalibrationPass | 66.66 | 3,410 | 103.1 | `cf149e174bc1` |
| r2 | none / 1024 | CalibrationPass | 64.77 | 3,354 | 105.0 | `0ddf925f2473` |

f1 failed before the barrier because the 2026-10-26 sandbox proof no longer matched `qwen38_n1_process.py` (changed
by `91d15441`). The 2026-10-31 proof and its approval templates differ from the old ones only in
`sandbox_proof_sha256`.

## Best valid configuration: provisional

By the declared rule (fastest `CalibrationPass` makespan; the two fastest within 5% repeated, alternating), the best
configuration is `--load-mode none --ubatch-size 1024` (median 65.22 s) over `auto/1024` (median 67.73 s). This is
**not separated beyond noise**: single-setting repeat spread is 1.4–3.2%, the ubatch choices rest on single runs whose
gaps (2–4%) are inside that spread, the ranking reverses per completion token (auto/1024 19.50/19.55 ms per token;
none/1024 20.39/19.31), and `none/512` (the matched peer arm's own load mode and ubatch) was not run.

**Carried into Q4-DS4-RACE:** re-sweep the peer on this file with repeated runs of `none/1024`, `auto/1024` and
`none/512` before fixing its best valid configuration.

## Per-stage profile (different artifacts; descriptive only)

Peer run r2 against hf2q H1 on its own artifact. Both are single runs; extractor `$V/q4-ds4/profile_extract.py`
(sha256 `b643d9ed…`), output `$V/q4-ds4/stage-profile-comparison.{json,txt}` (json sha256 `5830002a…`).

| Stage | Peer (ds4 file) | hf2q (own artifact) |
|---|---|---|
| Q5 verdict / agents passed | CalibrationPass / 4 | CalibrationPass / 4 |
| Makespan | 64.77 s | 102.96 s |
| Prompt / completion tokens | 38,738 / 3,354 | 39,697 / 3,698 |
| Cached-token share | 83.6% | 83.5% |
| Decode tok/s per stream: median / time-weighted | 16.81 / 16.36 | 10.83 / 11.94 |
| Decode lane seconds (sum over requests) | 205.0 | 309.7 |
| Prefill tok/s per request (median) | 125.1 | 74.2 |
| Prefill lane seconds (sum) | 42.4 | 72.8 |
| First content frame, median / p95 (first reasoning or semantic frame, per engine) | 1.59 / 2.12 s | 1.99 / 3.66 s |
| Streamed turns with a reasoning frame | 4 of 12 | 12 of 12 |
| Semantic TTFT | **not comparable** (1.72 s median) | **not comparable** (8.11 s = turn end) |
| SSE cadence | 5 semantic events per streamed turn; gap median 213 ms, p95 1,169 ms | 1 semantic event per streamed turn, emitted at turn end (median 0.009 s before the terminal frame); no gaps |
| Batching / scheduling | 3.78 busy slots per decode call (900 calls); 56 of 65 one-second samples at 4 busy; 0 deferred | 128 joint decode turns: 74 × 4 rows, 33 × 3, 21 × 2; one turn with serial leftovers |
| Dispatch | decode graph reused 800 times (final cumulative counter), 0 reallocations; GPU busy/idle Unknown | 86 command buffers per token (H2) |
| Prompt cache | 81 checkpoints created, 0 restored, 0 forced full re-processing | — |
| MoE / attention split | **Unknown**: no built-in per-operation GPU timer; no separate peer build and no frame capture (operator decision 2026-11-03) | one serial first decode step (H2): 28.6 ms = attention 19.6 + FFN 9.1 ms |
| Peak host used | 105.0 GiB | 118.1 GiB |

**Where the gap is.** The peer finishes 37.1% sooner on these runs. Per agent lane (sums ÷ 4), hf2q spends about
26.2 s more in decode and 7.6 s more in prefill, out of a 38.2 s makespan gap. **Decode is the largest contributor**:
the per-stream decode rate (0.73× time-weighted, 0.64× median) accounts for about 19–21 s per lane depending on the
token base, and hf2q's 10% larger completion volume (greedy divergence) for most of the rest. Prefill is second
(0.59× median per request). Semantic TTFT is not a contributor that can be read from this data: hf2q's one semantic
event per streamed turn arrives at turn end, so its "TTFT" is turn completion; on first content frame hf2q is 0.4 s
later at the median.

**Artifact effect (cannot be separated here).** Bytes read per token, ds4 file over hf2q artifact (GGUF tensor
extents; non-expert tensors except `token_embd`, plus routed experts × distinct experts read / 256):
- decode, 1 row (6 experts): 9.57 / 8.94 GB = **1.07** (the ds4 file reads about 7% more: Q8_0 shared experts and F16
  compressor/indexer outweigh its smaller IQ2_XXS experts);
- decode, 4 rows (about 23 distinct experts per layer): 14.74 / 15.60 GB = **0.95**;
- prefill (all experts): 86.7 / 107.4 GB = **0.81**.

Kernel cost per byte also differs by quantization (IQ2_XXS vs Q2_K/Q3_K), so the decode gap is not attributable to
bytes alone. The same-file attribution is Q4-DS4-RACE.

**Distortion.**
- Single run per arm; greedy output diverges between runs (peer completion tokens 3,221–3,529 across six runs, ±5%).
- Per-request prefill and decode seconds are lane times that overlap under four-way concurrency on both engines; they
  are compared as lane totals, never as wall time.
- Instrumentation, unquantified: the peer ran at `--log-verbosity 4` with 1 Hz `/metrics` and `/slots` polling
  (about 1 ms per poll); H1 ran the debug joint-decode trace filter.
- H2 is one first decode step on the serial graph (joint decode refuses the stage profile); its split is not the
  four-row joint split, and its 138 s makespan is not compared.
- The last peer `/metrics` poll was taken with one request still processing, so `/metrics` totals are slightly short
  of the log totals used above.
- Different artifacts, as above.

## Evidence

All under `$V/q4-ds4/`:
- `runs/peer-{f1,f1b,s1,s2,s3,r1,r2}-*/` (Q5 receipt, `server.stderr`, client summaries), each with sibling
  `.metrics.jsonl`, `.profile.{json,txt}` (regenerated with extractor `b643d9ed…`, schema
  `hf2q.q4-ds4.stage-profile.v2`), `.identities.txt`, `.host-{before,after}.txt`;
- `runs/hf2q-trace-h1-v0129-trace-20261103T230844Z/`, `runs/hf2q-stage-h2-v0129-stage-20261103T231216Z/`;
- `stage-profile-comparison.{json,txt}`, `fit-preflight.json` (pre-run memory estimate: 6.72 GiB KV predicted,
  7,095 MiB KV buffers observed), `run_q4_ds4.sh`, `poll_peer.py`, `profile_extract.py`, `cases/`;
- `review-fit-v1.md` (independent review).

Open: independent patch review of the runs' final sources.
