# ADR-063 D2 (#365): current pretrained drafters at four slots — result

Date: 2026-11-02. Host: M5 Max 128 GB (`runs-v2/host-sysctl.txt`). Status: primary,
abliterated (A) and qualification complete; non-repetitive code (N) incomplete
(see §6); long context (LC) partially complete (see §7).

`$E` = `/Users/robert/.local/state/hf2q/campaigns/adr063-preflight-20261023`;
`$D2` = `$E/d2-v1`. Every path below is relative to `$D2` unless absolute. The
plan, frozen inputs and pre-registration are `$D2/README.md` (v2 and v3
sections; sha256 at the time of writing `e1b6742a…2669`) and the two decision
comments on #365 (2026-11-02T01:53Z pre-run declaration; 2026-11-02T09:55Z
deviation disclosure).

## 1. Decision

**No pretrained drafter qualifies as successor scope. D2 closes negative for
DFlash2 and DSpark at four slots on Qwen3.8 27B.**

- **DFlash2** (official z-lab drafter): on its own engine TensorFold (T2) it
  completes 4/4 tasks with passing final tests in 3/3 runs, but its median
  makespan is 90.7 s vs H1 82.3 s, so it ranks below H1 in the primary group
  (§4). On the peer (P2) it corrupts output at four slots and is negative in
  qualification (§3). In LC it is the slowest passing arm (1034.5 s vs H1
  595.5 s, pass 1; §7).
- **DSpark** (Red Hat speculators): in the primary group P3 is arm-negative
  (one agent broke the six-step workflow; 0/4 completion, run dropped from later
  passes per the pre-registered rule; §4, §8). It ranks above H1 only on the
  abliterated target (A, 71.4 s vs 93.1 s), and the rule requires the primary
  rank, so A cannot make it successor scope.
- **Native MTP + history (H1)** does not beat H0 (speculation off) at N4:
  82.3 s vs 81.4 s median in the primary group, consistent with D1's measured
  Loss for `auto` at N4 (`docs/research/adr063-d1-speculation-qualification.md`
  §7, r = 1.020, CI 1.0095–1.0308).
- **Observation, not a drafter verdict:** the peer's own MTP head (P1) is the
  fastest primary arm (77.7 s median, 4/4 completion, 3/3 Q5 pass) but runs with
  2.00 GB worst memory headroom, and in LC it was stopped by Q5 for memory
  (0.33 GB headroom). P1 is the target's own MTP head, not a pretrained
  external drafter, so it is outside D2's drafter question; it is recorded for
  successor work on hf2q's MTP path at N4 (what the peer's MTP does at four
  slots that hf2q's `auto` does not).

Per the rule, no forced winner: the A ranking (P1 > P3 > P0 > H1 > H0) is
recorded but does not decide.

## 2. Setup and frozen inputs

Workload: vanilla Qwen3.8 27B target, 4 slots × 262144 tokens, the canonical
four-agent coding client; every run is one supervised Q5 calibration
(`scripts/four_agent/q5_calibration.py`). Runners:
`/opt/hf2q-worktrees/run-harness-9b7c6d24` at `005c9707` (qualification),
`/opt/hf2q-worktrees/run-harness-188c277e` at `188c277e` (primary, A, N),
`/opt/hf2q-worktrees/run-harness-d60d38de` at `d60d38de` (LC,
`--workload long-context-v1`, PR #551). Base cases `q5_cases/qwen38.json` and
`peer-qwen38.json` are byte-identical at 005c9707 and 188c277e (README v2
preamble). Every script re-hashes the frozen inputs before running (`lib_v2.sh`;
`run_extra_n_rerun.log` shows the `ok <sha256>` lines).

| Input | Identity (sha256 unless noted) | Source |
|---|---|---|
| vanilla target `Qwen3.8-27B-vanilla.gguf` | `cf0eb0103c1ffe7b3702670aa794a669bafbe328b95e892268305dd87c2c30d7` | README v2 frozen inputs |
| abliterated target `qwen38-abliterated-sft-hf2q-q4_k_m.gguf` | `1ee55c653644d6f645c6b2f39fc56a3ce28093620fd34dd43678875f348f2e1a` | same |
| TensorFold target `TensorFold/Qwen3.8-27B-MLX-4bit@22d8d538` `config.json` / `chat_template.jinja` (hf2q Qwen3.8 template) | `14b65a0e…85e85` / `c3cf9e34…81041` | same |
| DFlash2 drafter (T2) `z-lab/Qwen3.8-27B-DFlash2@50307d4c4cde6860d4eee73e2547cd786fe8e8a4`, `model.safetensors`, loaded `--drafter-bits 4` | `67fc76d68dc5a9415511a4f394ef744d67510cd20e93b37cc2cc7d28e4bab65c` | same |
| DFlash2 GGUF (P2) `z-lab/Qwen3.8-27B-DFlash2-GGUF@2d9571f8`, BF16, 3,860,293,216 B | `26d47ca20ab07688327a63d912acad222d924eaaa92a980cc488de3c67e736bc` | README "Files and hashes" |
| DSpark drafter (P3) Red Hat speculators `87ca2fdc`, converted to BF16 GGUF with the peer converter at `e86f5fd1`, 3,987,938,944 B; input safetensors `6cf6f33f…1781` | `de6244609645fe306d951fe2aad183d8290b9de35aa4ce6393585229d1be5aba` | `drafters/convert-dspark.receipt.json` |
| hf2q binary `q3-experiment-bin-38485a23…/hf2q` | `fbf6bfcf10d3ca0ade57308d75337221eae7509a1356e97f0a675dfa5a49f714` | README v2 |
| peer server binary, build `b128-e86f5fd` | `4b604952a8ac42121974fec1124b34bdeb345383450f0c17c5a20a4d16458c6b` | same |
| `tensorfold-native` 1157ea5b (1.2.0) | `6a52fd9153dd5559b3f3d28d2194ccaeb78920ceb641454b97eeaa0850867064` | same |
| sandbox proof `owned-cleanup-sandbox-capability-20261031T222313Z/proof.json` | `8b2866842913c206b9b801ced3c86397b5ccde7631b8422fb2166bdbe26ea589` | same |
| cases receipt `cases/v2-cases-receipt.json` (T0/T2 and 19 extra cases) | `de5a8d9ba8654c286e0b40bffb7dcd37e8f2f65e3a2ca0b70a9d43c18f9d72bf` | same |
| LC cases receipt `cases/long-context/cases-receipt.json` | `b8ec7bae8d4d430df20b112f09a9f0f60226db14ff3587d10e14a9538baa840b` | README v3 |
| `extract.py` | `a88f99e1aad64c9bfdf9bf5ad179c0efa65853c702385e40d58f78119d680cc6` | README v3; re-hashed 2026-11-02 |

Drafter metadata (README "Drafter GGUF metadata"): DFlash2 GGUF is
`general.architecture=dflash`, 81 tensors, no `markov_w1.weight`,
`target_layers [6,20,34,48,62]`, `block_size 8`, sliding window 2048. DSpark is
`dflash`, 62 tensors, `markov_w1.weight` present, `target_layers
[4,12,20,28,36,44,52,60]`, `sample_from_anchor`, confidence head. Tokenizer and
chat template are the target's (the drafters share the target vocabulary; the
TensorFold template is the hf2q Qwen3.8 template, hash above).

Summary-file hashes at the time of writing: `qualify/summary.tsv`
`f0ec234d…5ca`, `runs-v2/summary.tsv` `81f87ff0…cb3`,
`extra-runs/A/summary.tsv` `a0351e52…3f2`, `extra-runs/N/summary.tsv`
`b35c02ef…56a`.

## 3. Arms and qualification

| Arm | Engine / drafter | Role |
|---|---|---|
| H0 | hf2q 38485a23, `QWEN38_SPECULATION=off` | hf2q baseline |
| H1 | hf2q 38485a23, `auto` (native MTP + history) | current MTP/history (the bar) |
| P0 | peer, `--spec-type none` | peer baseline |
| P1 | peer, `draft-mtp` (target's own MTP head, n-max 3) | peer MTP (not an external drafter) |
| P2 | peer, `draft-dflash` (DFlash2 GGUF, n-max 7) | DFlash2 on the peer |
| P3 | peer, `draft-dspark` (DSpark BF16, n-max 7) | DSpark on the peer |
| T0 | TensorFold, no drafter | TensorFold baseline |
| T2 | TensorFold + official DFlash2 (4-bit) | DFlash2 on its official engine |

Qualification (`run_qualify.sh`, one N4 run each at `--log-verbosity 4`,
`qualify/summary.tsv`):

| Arm | Run | Result | Evidence |
|---|---|---|---|
| P0 | `qualify/01-P0-none-20261101T030309Z` | qualified | `qualify/01-P0-none-*.checks.txt` |
| P1 | `qualify/02-P1-mtp-20261101T030522Z` | qualified; `ExactTranscripts` vs P0, 52 groups, 0 differences | `qualify/02-P1-mtp-*.transcripts-vs-P0.json` |
| P2 | `qualify/03-P2-dflash2-20261101T030656Z` | **negative-run** (Q5 `ClientFailed`, client exit 1) | below |
| P3 | `qualify/04-P3-dspark-20261101T030804Z` | qualified; `extract_layers = [4, 12, 20, 28, 36, 44, 52, 60]`, SWA KV 8704 cells; `ExactTranscripts` vs P0, 52 groups, 0 differences | `qualify/04-P3-dspark-*.checks.txt`, `*.transcripts-vs-P0.json` |

**P2 corrupted output.** At four slots the peer's DFlash2 path emitted tool-call
markup inside the reasoning channel:
`qualify/03-P2-dflash2-20261101T030656Z/client/agent-3/turn-4/response.bin`
contains `"reasoning_content":"Both port>\n</function>\n</tool_call>"`, and
agent 3 then failed with `exactly one live call required`
(`client/agent-3/summary.json`; `extract.py classify` →
`task-failed:Invalid: exactly one live call required`). Greedy verification
should never change target output, so this is a drafter-path correctness
failure on the peer at N4, not a speed result. P2 was excluded from all later
groups; DFlash2 is measured on its official engine instead (T2).

## 4. Primary results (vanilla target, 3 interleaved passes)

Order `H1 H0 P0 P1 P3 T0 T2 | T2 T0 P3 P1 P0 H0 H1 | H1 H0 P0 P1 P3 T0 T2`
(`run_primary_v2.sh`). Command (read-only, launches nothing):

```sh
python3 -B $D2/extract.py table $D2/runs-v2 --json $D2/runs-v2/d2-extract.json --md $D2/runs-v2/d2-extract.md
```

Per-arm aggregates (`runs-v2/d2-extract.md`; re-run 2026-11-02 with identical
output). Completion = agents in `ProtocolCompleteAwaitingIndependentPatchReview`;
quality = agents whose last `run_tests(all)` passed; headroom = min free+cached
minus Q5 effective floor (worst run); makespan =
`own_output_four_task_makespan_seconds` (median); acc/verify = accepted per
verify step (emitted per verify step = this + 1 [INFERENCE, extract.py
docstring]); tok/decode-s = completion tokens per second of in-request decode
time (draft + verify + recovery); tok/makespan-s = completion tokens per
makespan second (net, four slots).

| Arm | Runs | Q5 pass | Completion | Quality | Worst headroom GB | Median peak GB | Median makespan s | Acc/draft | Acc/verify step | Tok/decode-s | Tok/makespan-s |
|---|---|---|---|---|---|---|---|---|---|---|---|
| H0 | 3 | 3 | 1.00 | 1.00 | 50.94 | 54.93 | 81.4 | – | – | 13.80 | 41.90 |
| H1 | 3 | 3 | 1.00 | 1.00 | 46.12 | 59.71 | 82.3 | 0.72 | 2.15 | 13.62 | 41.78 |
| P0 | 3 | 3 | 1.00 | 1.00 | 12.94 | 99.01 | 118.9 | – | – | 16.32 | 57.12 |
| P1 | 3 | 3 | 1.00 | 1.00 | 2.00 | 106.10 | 77.7 | 0.83 | 2.46 | 26.48 | 86.26 |
| P3 | 1 | 0 | 0.00 | 0.75 | 5.35 | 109.43 | – | 0.53 | 3.62 | 17.68 | – |
| T0 | 1 | 0 | 0.00 | 0.75 | 6.42 | 94.18 | – | 0.39 | 0.18 | 28.18 | – |
| T2 | 3 | 3 | 1.00 | 1.00 | 2.83 | 96.63 | 90.7 | 0.52 | 4.04 | 66.78 | 70.39 |

Per-run makespans (s): H0 83.3/80.3/81.4; H1 83.4/82.1/82.3; P0
119.5/118.9/118.6; P1 77.7/78.2/76.2; T2 90.1/90.7/92.5 (`runs-v2/d2-extract.md`
per-run table; runs `01…21-*` in `runs-v2/`). Tool calls: 24 in every passing
run, 22 in the two failing runs. Max prompt: 2368 (hf2q), 2804–2806 (peer),
3584–3615 (TensorFold) tokens.

Speculation cost notes:
- H1 drafted only 36–63 tokens per run (`runs-v2/*-H1-auto-*.metrics/last.prom`),
  so native MTP/history barely engages at N4 and its decode rate (13.62 tok/s)
  equals H0's (13.80).
- T2 has the highest per-request decode rate (66.78 tok/s, 4.04 accepted per
  verify step) but not the best net four-slot makespan: the decode-rate
  headline does not decide (README v2 decision rule).
- P1 accepts 0.83 of drafted tokens at 2.46 per verify step and doubles the
  peer's decode rate over P0 (26.48 vs 16.32 tok/s).

**Order** (`extract.py`, lexicographic: completion, quality, memory tier,
median makespan): **P1 > H0 > H1 > T2 > P0 > P3 > T0.** P1 and H0 rank above
H1; T2, P0, P3, T0 rank below.

## 5. Abliterated/grafted target compatibility (A)

Target: abliterated/SFT GGUF (`1ee55c65…`); same tokenizer, template and
metadata as vanilla except `general.name`; its 15 MTP-head tensors are
byte-identical to vanilla, 64 of 145 compared trunk tensors differ
(`extra/abliterated-target-compat.json`). 10 runs, `H1 H0 P0 P1 P3 | P3 P1 P0
H0 H1` (`extra-runs/A/summary.tsv`). TensorFold has no MLX checkpoint of this
target, so T0/T2 are unavailable here.

`python3 -B $D2/extract.py table $D2/extra-runs/A`:

| Arm | Runs | Q5 pass | Completion | Quality | Worst headroom GB | Median peak GB | Median makespan s | Acc/draft | Acc/verify step | Tok/decode-s | Tok/makespan-s |
|---|---|---|---|---|---|---|---|---|---|---|---|
| H0 | 2 | 2 | 1.00 | 1.00 | 52.01 | 55.06 | 94.8 | – | – | 13.15 | 41.35 |
| H1 | 2 | 2 | 1.00 | 1.00 | 47.32 | 59.84 | 93.1 | 0.79 | 2.36 | 13.47 | 41.97 |
| P0 | 2 | 2 | 1.00 | 1.00 | 14.12 | 98.75 | 78.1 | – | – | 15.69 | 46.06 |
| P1 | 2 | 2 | 1.00 | 1.00 | 2.72 | 106.40 | 53.1 | 0.78 | 2.30 | 25.92 | 69.10 |
| P3 | 2 | 2 | 1.00 | 1.00 | 5.18 | 109.21 | 71.4 | 0.48 | 3.22 | 16.74 | 48.80 |

Order: **P1 > P3 > P0 > H1 > H0.** Every drafter stays compatible with the
modified trunk (acceptance 0.48 for DSpark vs 0.53 on vanilla; MTP 0.78 vs
0.83). The A makespans are not comparable with the primary ones: the
abliterated target's agents produce different trajectories (peer arms are
faster here, hf2q arms slower), so A answers compatibility, not ranking.

## 6. Non-repetitive code (N) — incomplete

Cases `cases/extra/N{1,2}-{H0,H1,P0,P1,P3,T0,T2}.json` (task instances 1
`parse_privileged_port` and 2 `parse_vlan_id`; 14 pre-registered runs).
`extra-runs/N/summary.tsv`:

| Run | Arm | Q5 | Completion | Quality | Makespan s | Note |
|---|---|---|---|---|---|---|
| `01-N1-H1-20261102T054653Z` | H1 | CalibrationPass | 4/4 | 4/4 | 88.2 | drafted 39, accepted 19 |
| `02-N1-H0-20261102T054830Z` | H0 | ClientFailed | 0/4 | 3/4 | – | `task-failed:Invalid: initial six-step workflow not followed` (agent 0) |
| `03-N1-P0-20261102T100003Z` | P0 | PreflightRefused | – | – | – | `competing runtime(s): 35678:build, 60983:build` (`run_extra_n_rerun.log`) |

N1-H0 halted `run_extra.sh` under the original halt rule (§9). The resumed
`run_extra_n_rerun.sh` then hit Q5's quiet-host preflight: two `build`
processes were running on the host at 10:00Z, Q5 refused to launch, and the
script halted as an infrastructure failure (correct under the v3 policy). 11
pre-registered N runs (N1-P1 … N2-H1) are therefore **unavailable**. The script
is resumable and skips recorded cases:

```sh
bash $D2/run_extra_n_rerun.sh 2>&1 | tee -a $D2/run_extra_n_rerun.log   # on a quiet host
python3 -B $D2/extract.py table $D2/extra-runs/N
```

Effect on the decision: none for the drafters. The rule makes N a veto on a
drafter that already ranks above H1 in the primary group; neither DFlash2 (T2)
nor DSpark (P3) does. N would matter only for the P1 observation.

## 7. Long context (LC) — in progress

`long-context-v1`: canonical parse_port task plus a 150 KB read-only Rust
module; from turn 2 every request carries about 40k wire-estimated tokens
(measured max prompt 41,732–42,168 tokens). Arms H1 H0 P0 P1 T2 (P3, T0
omitted: README v3). One H1 fit run, then two passes `H1 H0 P0 P1 T2 | T2 P1 P0
H0 H1` into `long-context-runs/passes/` (`long-context-runs/summary.tsv`).

Fit run `long-context-runs/fit/00-LC-H1-20261102T100038Z`: CalibrationPass, 4/4,
4/4, makespan 635.9 s, headroom 23.82 GB (not pooled).

Pass 1 (`extract.py table long-context-runs/passes`, runs 01–05, as of
2026-11-02 ~11:10Z):

| Run | Arm | Q5 | Completion | Quality | Makespan s | Headroom GB | Peak GB | Acc/draft | Emitted/verify step | Tok/decode-s | Cached / uncached prompt tokens |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `01-LC-H1-20261102T101123Z` | H1 | CalibrationPass | 4/4 | 4/4 | 595.5 | 32.76 | 62.05 | 0.61 | 2.83 | 9.09 | 660,884 / 167,606 |
| `02-LC-H0-20261102T102128Z` | H0 | CalibrationPass | 4/4 | 4/4 | 584.5 | 38.21 | 57.27 | – | – | 8.50 | 660,884 / 167,606 |
| `03-LC-P0-20261102T103122Z` | P0 | CalibrationPass | 4/4 | 4/4 | 621.3 | 0.58 | 107.90 | – | 1.00 | 5.23 | 627,860 / 205,959 |
| `04-LC-P1-20261102T104157Z` | P1 | **Stopped** (memory) | 0/4 | 1/4 | – | 0.33 | 112.62 | 0.85 | 3.53 | 6.56 | 254,161 / 161,573 |
| `05-LC-T2-20261102T104912Z` | T2 | CalibrationPass | 4/4 | 4/4 | 1034.5 | 3.41 | 96.67 | 0.42 | 5.66 | 40.49 | 165,301 / 665,227 |

(cached/uncached prompt tokens from the per-run lines in `run_long_context.log`.)
LC-P1 is `arm-negative memory-refused` and dropped from pass 2
(`run_long_context.log`). T2 decodes fastest per request but prefills about 4×
more uncached tokens than the hf2q/peer arms, which is consistent with its
1034.5 s makespan [INFERENCE: prefix-cache reuse on TensorFold at N4 is weaker;
not separately measured].

Pass-1 order (completion, quality, memory, makespan): **H0 > H1 > P0 > T2 >
P1**; T2 ranks below H1.

<!-- TODO(LC): when long-context-runs/summary.tsv shows pass 2 finished
(runs 06-LC-T2 … 10-LC-H1), replace the pass-1 table with the per-arm aggregate
from the passes only (the fit run must not be pooled):

  python3 -B $D2/extract.py table $D2/long-context-runs/passes \
      --json $D2/long-context-runs/passes/d2-extract.json \
      --md $D2/long-context-runs/passes/d2-extract.md
  (cd /opt/hf2q-worktrees/run-harness-d60d38de && \
   python3 -B scripts/four_agent/native_peer_tensorfold_audit.py <LC-T2 runs>)

and restate the LC order. -->

## 8. Unavailable and negative arms

| Arm / group | Status | Why | Evidence |
|---|---|---|---|
| P2 (peer DFlash2), all groups | negative in qualification | corrupted tool-call output at N4; agent 3 "exactly one live call required" | §3 |
| P3 (peer DSpark), primary | arm-negative after run 1 | agent broke the six-step workflow; 0/4 completion, 3/4 final tests | `runs-v2/05-P3-dspark-20261102T050850Z`; `extract.py classify` → `task-failed:Invalid: initial six-step workflow not followed` |
| T0 (TensorFold baseline), primary | arm-negative after run 1 | same task failure | `runs-v2/06-T0-tensorfold-20261102T051027Z`, same classify result |
| T0/T2, group A | unavailable | no MLX checkpoint of the abliterated target | README v2 |
| N1-P1 … N2-H1 (11 runs) | unavailable | halted: H0 task failure under the original rule, then Q5 quiet-host preflight refusal | §6 |
| LC-P1 | arm-negative | Q5 stopped the run for memory (0.33 GB headroom) | §7 |
| LC-P3, LC-T0 | not run (by design) | P3 already below H1 on completion; T0 is not a candidate | README v3 |

The P3 and T0 primary failures are task failures (an agent violated the
client's workflow), not memory or infrastructure; the same failure class hit
reference arm H0 in N1, so it is not drafter-specific evidence by itself. The
pre-registered drafter rule (drop after a failure) still applies and is kept.
P3 passed both its qualification run and both A runs.

## 9. Deviations (disclosed)

- **N halt policy** (#365 comment 2026-11-02T09:55Z; README v3 "Failure policy
  from v3 on"): `run_extra.sh` halted block N on any nonzero exit of a reference
  arm, so N1-H0's task failure cost 12 pre-registered runs. From v3 a reference
  task failure is `task-negative` and the block continues; infrastructure or
  memory failures still halt; drafter/TensorFold arms stay drop-on-failure.
  Primary and A are unaffected (no reference failures there).
- **Long context added in v3** (PR #551, runner d60d38de) after it was first
  recorded unavailable at 188c277e (`extra/long-context-unavailable.json`).
- **Runner drift:** qualification (005c9707) and later groups (188c277e,
  d60d38de) use different harness commits; qualification transcripts are not
  compared with v2 runs (README "Open risks").
- **Drafter decision ranking** uses the v2 lexicographic rule (memory as a Q5
  tier), which supersedes the v1 "beats H1 on all four" wording; both lead to
  the same result here because no drafter beats H1 on makespan in the primary
  group.

## 10. Reproduction

```sh
D2=/Users/robert/.local/state/hf2q/campaigns/adr063-preflight-20261023/d2-v1
bash $D2/run_qualify.sh                 # qualification (runner 9b7c6d24)
bash $D2/run_primary_v2.sh              # primary (runner 188c277e)
bash $D2/run_extra.sh                   # A, then N
bash $D2/run_extra_n_rerun.sh           # remaining N (resumable)
bash $D2/run_long_context.sh            # LC (runner d60d38de)
python3 -B $D2/extract.py table $D2/runs-v2
python3 -B $D2/extract.py table $D2/extra-runs/A
python3 -B $D2/extract.py table $D2/extra-runs/N
python3 -B $D2/extract.py table $D2/long-context-runs/passes
```

Drafter files are reference-only benchmark inputs (operator decision
2026-10-31); hf2q never ships or serves them.
