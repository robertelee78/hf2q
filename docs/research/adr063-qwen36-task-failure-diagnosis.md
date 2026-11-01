# ADR-063: Qwen3.6 four-agent coding failure, diagnosis

Status: measured, 2026-10-30, M5 Max 128 GB. Outcome: the failure comes from the
Qwen3.6-35B-A3B weights, not from hf2q.

## Question

Qwen3.6 fails the canonical four-agent coding workload most of the time. In the
24-instance paired campaign, hf2q passed 5 instances and the peer 4 (sign test
p = 0.77). Matching the peer does not by itself rule out hf2q. Both arms shared
the hf2q-converted GGUF, the chat template, the sampling settings and the
harness. This record tests each shared factor separately.

## Failure signature

All 20 failing hf2q agents and all 22 failing peer agents fail the same way:

1. The first `write_file` contains `let v: u16 = s.parse().map_err(|e| e.to_string())?;`.
2. `run_tests` fails with E0282: the closure parameter's type is still
   unknown when `e.to_string()` needs it.
3. The model then writes a fix, but the strict six-step workflow allows no
   repair turn, so the agent fails.

Every passing agent compiled on its first try, using `|_| …` or
`|e| format!("…{}", e)`. The operator kept the strict workflow; it is the
canonical workload.

## Discriminators

Each campaign below has 24 instances, one four-agent cohort each, run under the
Q5 supervisor on hf2q `0b959107` unless stated otherwise.

| Arm | Instances passed |
|---|---:|
| hf2q Q4_K_M; repetition penalty 1.05, tool-thinking 512 (canonical) | 5 |
| Peer, same GGUF and settings | 4 |
| hf2q, repetition penalty 1.0 | 8 |
| hf2q, tool-thinking budget 2048 | 5 |
| hf2q Q8_0, converted by hf2q from the same source revision | 3 |

Write-turn replays use the recorded `write_file`-turn requests: agents 0 and 1
of every instance, 48 prompts in total. The count is the number of outputs
containing `|e| e.to_string()`:

| Arm | Count |
|---|---:|
| hf2q Q4_K_M, greedy | 22/48 (21 failing; one hit uses a compiling form) |
| Peer Q4_K_M, greedy (its own transcripts) | 31/48 |
| hf2q Q8_0, greedy | 34/48 |
| hf2q Q4_K_M, vendor sampling (temp 0.6, top_p 0.95, top_k 20, 4 seeds) | 86/192 (45%) |
| Unconverted BF16 safetensors in an independent reference implementation, greedy | 41/48 |

Fork logprobs: across all 233/233 greedy fork positions, the chosen token is the
raw top-1. Logprobs are captured before the repetition penalty is applied, so
the penalty never changed the pick. The top-1 minus top-2 margins have medians
of 0.58 to 0.77 nat. At the closure parameter, the signed `e` − `_|` medians
are −0.09 nat (hf2q Q4), +0.34 (hf2q Q8) and +0.28 (peer); |e − _| is below
0.25 nat in 9, 11 and 11 of 48 prompts and below 0.10 nat in 4, 4 and 3. From
Q4 to Q8 on identical prompts the margin moves by a median 0.76 nat and
changes sign in 21/48 prompts, so quantization often flips this fork. The body
fork (`e.to_string` vs `format`, after `|e|`) is firm: medians 1.03 to 1.98
nat, minimum 0.12.

The direction settles attribution: on the same 48 prompts the idiom rate rises
with precision, Q4 22 → Q8 34 → BF16 41 (independent implementation); the peer
shows 31/48 on its own transcripts. hf2q's quantized numerics reduce the
failure rather than cause it.

No formal §3.4 fork classification was run: the peer replay used the peer's own
transcripts, not byte-identical token IDs, so `q1_decision_policy.py fork` was
never executed. These forks are not classified NearTie or NumericSuspect. The
numeric policy's §5.2 step 1 ("Fork first") is superseded by step 4 (the paired
campaign) plus the BF16 reference.

The reviewed template is byte-identical to the vendor `chat_template.jinja` at
the pinned revision (sha256 `e84f32a2…`). Two independent renderers, hf2q's and
the peer's, both produce the failure.

## Conclusion

| Shared factor | Verdict | Evidence |
|---|---|---|
| Engine | excluded | hf2q matches the peer, p = 0.77 |
| Repetition penalty | excluded | never changed the pick; 8/24 with no penalty |
| Thinking cap | excluded | median 140 reasoning tokens; 5/24 at 2048 |
| Quantization | excluded | Q8_0 no better (3/24); idiom rate rises with precision (22 → 34 → 41 of 48) |
| hf2q converter | excluded | the independent BF16 reference produces the idiom 41/48 times |
| Template | excluded | byte-identical to the vendor template |
| Greedy vs vendor sampling | excluded | 45% vs 46% |

The non-compiling idiom is a property of the model on this task. Under the
retained strict workload, Qwen3.6 is BelowFloor on every engine and precision
tested. Its Q3/Q4 verdict is **Failed** on task quality, which is not
attributable to hf2q. Retained hf2q defects found during this work were fixed
separately in #461.

## Reproduction

`E` is the campaign root
`/Users/robert/.local/state/hf2q/campaigns/adr063-preflight-20261023`. The
canonical paired campaign is `$E/quality-qwen36-peer-exp0b95-20261030T174020Z`.
The diagnostic scripts live outside the repository under `$E/qwen36-fork-v1/`:

- `replay.py`: logprob fork replay through the OpenAI API
- `run_fork.sh`: one memguarded server per arm
- `ref_mlx.py`: BF16 reference
- `chain.sh`: the sequential order of the runs above

The summary is in `$E/qwen36-fork-v1/diagnosis-summary.json`. The campaign
manifests are in `$E/quality-qwen36-*`.

## Disclosures

- An earlier paired campaign, `$E/quality-qwen36-peer-20261030T173012Z`
  (subject binary `f53f60ed…`), was abandoned after 12 launched runs; 11 failed
  on both arms and its manifest still says Running. Policy §3.5 forbids reruns.
  It cannot bias the result toward failure, but it is not part of the verdict.
- The first Q8 attempt (`…212129Z`) halted on `readiness_mismatch` before any
  client launched (the vision projector had loaded). The rerun (`…212204Z`) is
  not in `chain.sh` or `chain.log`.
- The vendor-sampling and BF16 commands are not in `chain.sh`.
- The Q8_0 GGUF and the BF16 source were deleted. The Q8 sha is kept in the
  campaign receipts. Reproducing either arm requires re-downloading the source
  at revision 995ad96e and reconverting.
