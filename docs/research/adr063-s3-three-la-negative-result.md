# Three linear-attention layers per checked block — negative result

Isolated source `353cdc4ba9b94da1a3ccc64a2a20c90e92f20fc4`, optimized test binary
`00e2a58916ec00fcd349babd5dd7c7196dd0cb126d872da233f357cff5dc3226`,
public mlx-native 0.15.12. Thirty focused host controls, 42 strict-parser controls
and the bound authority controls passed before the actual model experiment.

The fixed Qwen3.8 diagnostic retains the previous model, C262144/N4 allocation,
owners [3,0], teacher-forced suffix, full cold preparations and fixed ABBA schedule.
It uses the distinct `qwen38-n2-three-la-checked-group-v1` profile and v3 artifacts:
32 terminal FFN anchors and 352 retained submissions, including 32 intermediate
FFNs. All actual statuses are checked; the 96 Aux submissions are outside the
chosen GPU timing stream. Wait-only intervals are explicitly labeled.

|Arm|Median of all ten timed samples|
|---|---:|
|Normal ordinary forward|263.884167 ms|
|Three-LA shared checked graph|277.5121255 ms|

Shared is **5.164371% slower** in this diagnostic. No sample filtering or cross-run
causal comparison is used. All 24 cold preparations, 12 paired comparisons, 1320
manifest coordinates, 260 cache roots and cross-layer chosen event orders passed.
The source's finite/bitwise assertions support the hash metadata; raw tensor bytes
were not exported. This is not a full agentic serving or AUTO-tail qualification.

Actual directory: local campaign `adr063-preflight-20261023`,
`s3-qwen-three-la-target-timing-actual-v1`. Supervisor: 387.092892 seconds,
754 observations, exit 0, child reaped and empty process group. Independent audit:
`/tmp/adr063-s3-three-la-actual-v1-independent-audit.md` and JSON. All 218 historical
roots matched full pre/post metadata digests; mutable I/O was 147.148 MB within
384 MiB; allowed thermal states and zero swap-growth controls held.

The candidate remains an isolated negative design spike. No production adoption,
other-family win, unique physical-fit proof or whole S3 acceptance is claimed.
S3 stays Designing with F2/Q1/Q5 predecessors unchanged. Whole Q5 memory admission
remains a critical predecessor; this failed performance result does not satisfy it.
