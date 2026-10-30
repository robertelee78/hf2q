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
| hf2q Q4_K_M, greedy | 22/48 |
| Peer Q4_K_M, greedy (its own transcripts) | 31/48 |
| hf2q Q8_0, greedy | 34/48 |
| hf2q Q4_K_M, vendor sampling (temp 0.6, top_p 0.95, top_k 20, 4 seeds) | 86/192 (45%) |
| Unconverted BF16 safetensors in an independent reference implementation, greedy | 41/48 |

Fork logprobs: across all 288 fork positions, the chosen token is the raw
top-1. Logprobs are captured before the repetition penalty is applied, so the
penalty never changed the pick. The margins are not near-ties. At the closure
parameter (`e` vs `_|`), the median margin is 0.58 to 0.77 nat. After `|e|`,
`e.to_string` beats `format` by a median of 1.0 to 2.0 nat.

The reviewed template is byte-identical to the vendor `chat_template.jinja` at
the pinned revision (sha256 `e84f32a2…`). Two independent renderers, hf2q's and
the peer's, both produce the failure.

## Conclusion

| Shared factor | Verdict | Evidence |
|---|---|---|
| Engine | excluded | hf2q matches the peer, p = 0.77 |
| Repetition penalty | excluded | never changed the pick; 8/24 with no penalty |
| Thinking cap | excluded | median 140 reasoning tokens; 5/24 at 2048 |
| Quantization | excluded | Q8_0 no better (3/24) |
| hf2q converter | excluded | the independent BF16 reference produces the idiom 41/48 times |
| Template | excluded | byte-identical to the vendor template |
| Greedy vs vendor sampling | excluded | 45% vs 46% |

The non-compiling idiom is a property of the model on this task. Under the
retained strict workload, Qwen3.6 is BelowFloor on every engine and precision
tested. Its Q3/Q4 verdict is **Failed** on task quality, which is not
attributable to hf2q. Retained hf2q defects found during this work were fixed
separately in #461.

## Reproduction

`E` is the campaign root `adr063-preflight-20261023`. The diagnostic scripts
live outside the repository under `E/qwen36-fork-v1/`:

- `replay.py`: logprob fork replay through the OpenAI API
- `run_fork.sh`: one memguarded server per arm
- `ref_mlx.py`: BF16 reference
- `chain.sh`: the sequential order of the runs above

The summary is in `E/qwen36-fork-v1/diagnosis-summary.json`. The campaign
manifests are in `E/quality-qwen36-*`.
