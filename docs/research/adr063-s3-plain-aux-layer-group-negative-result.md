# Three ordinary asynchronous attention boundaries — negative diagnostic result

The isolated source `686af5f40676736e22a5163d005047dd3c1e4762` deferred the selected
Ops1–4, SDPA and Ops6–7 submissions into each existing checked layer group.
It retained every actual command buffer through its healthy check; genuine CPU
reads, bisect boundaries and final head checks remained intact.

The optimized test binary SHA256 was
`78534cd5b9d0100c279a6230453af214f059f63adbb50d1582b39fb8127505e0`,
using public mlx-native 0.15.12. All 23 focused host controls and 12 runtime
authority controls passed before the actual run.

The fixed diagnostic used the selected Qwen3.8 artifact, full C262144/N4 cache
allocation, owners [3,0], fixed teacher-forced next-turn suffix and 24 full cold
preparations: two warm and ten timed samples per arm, without sample filtering.
This is a matched-input target-forward diagnostic, not an agentic serving trial.

|Arm|Median of all ten timed samples|
|---|---:|
|Normal ordinary forward|265.2012495 ms|
|Shared checked layer group with Aux deferral|299.798771 ms|

Shared was **13.045761% slower**. All 12 paired comparisons, 1320 state coordinates,
260 roots and 3456 chosen timing events passed their defined checks. The 96 Aux
checks are not represented as observed GPU timestamps. Host durations can include
queued upstream work; command-buffer duration sums are not GPU busy time or union.
No difference against an earlier run is presented as a causal speedup.

Actual evidence is retained locally under campaign `adr063-preflight-20261023`,
directory `s3-qwen-aux-layer-group-target-timing-actual-v1`. The supervisor recorded
389.827719 seconds, 758 observations, exit 0 and a reaped empty process group.
Independent audit: `/tmp/adr063-s3-aux-group-actual-v1-independent-audit.md` and JSON.
Both full historical metadata scans matched; mutable I/O stayed within 384 MiB;
operating limits, allowed thermal states and zero swap-growth checks passed.
These are operating observations, not proof of unique physical memory fit.

This failed performance candidate stays on its isolated design-spike branch.
No production adoption, full AUTO/cancellation/error proof, serving-speed win,
other-family win or whole S3 acceptance is claimed. S3 remains in Designing,
with F2/Q1/Q5 predecessors unchanged. The next source hypothesis reduces the
number of checked layer anchors while preserving all command-buffer health and
ownership checks; its correctness and performance require independent proof.
