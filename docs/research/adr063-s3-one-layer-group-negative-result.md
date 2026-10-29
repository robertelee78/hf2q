# ADR-063 S3: one-layer checked grouping did not improve target latency

Updated: 2026-10-28. Status: negative isolated spike; no production adoption.
This record belongs to `spike/s3-one-layer-checked-group`. The failed speed
hypothesis and its implementation remain outside `main`; this is not a PR or
an S3/F2 acceptance claim.

The proposed one-layer retained submission group preserved the tested target
state/output equality, but its shared target median was **326.1395835 ms**
versus **262.1835205 ms** for the ordinary target arm in the same run:
shared/normal **1.243936243124785**, or **24.3936% longer latency**. It did not
beat the ordinary producer on this deciding interval.

## Exact experiment and retained evidence

| Binding | Value |
|---|---|
| Source | `d8585a1025296bd22a1a7a9415d8f97615611d9e` |
| Release test-binary SHA256 | `4952db550a7dc6fa32bd3779b8454dc1d1990e0eea4a1eb2168512d4f10e2715` |
| Public dependency | mlx-native `0.15.12` |
| Timing profile | `qwen38-n2-one-layer-checked-group-v1` |
| Actual campaign directory | `adr063-preflight-20261023/s3-qwen-layer-group-target-timing-actual-v1` |
| Independent review artifact | `adr063-s3-layer-group-actual-v1-independent-audit.md` |
| Review SHA256 | `53ba67325889ddc576bdfc7f231e3376c16cc0dcd77e5d5bfabf124718882ac9` |
| Independent reconstruction JSON | `adr063-s3-layer-group-actual-v1-independent-audit.json` |
| Reconstruction SHA256 | `ea38188c7e9ac9c2b8102347b8b777adf52bf36c7aac66b1868a633ad5677936` |
| Strict protocol parser SHA256 | `6d2c28483a834bde96aca343810972c826df8b23af6b651b6367deb1e62eb7e5` |

The private campaign retains packet, composition, operating policy/review,
activation receipt, exact supervisor, build receipt, canonical input bindings,
both executable copies, raw stdout/stderr, samples/state records, process and
host observations, historical inventories and final receipt. Large artifacts
and local machine configuration are not committed. The independent audit read
retained metadata/source and rehashed bindings; it did not rerun the model or
independently read the complete float output payload.

## Within-run result

The actual release test executed one complete fixed schedule: two warm
executions per arm followed by five ABBA blocks, yielding ten timed samples
per arm and twelve matched pairs including warm executions. All samples were
retained; the medians above use all ten timed samples in each arm. No outlier
filter or favorable subset was selected.

Each of the 24 executions rebuilt the declared cold state for four owners in
one full-capacity cache. Shared target row order was `[3, 0]`; four configured
owners did not establish width-four shared target execution. The target
interval excluded cold preparation and state digest work. Inputs were fixed
teacher-forced suffix tokens, not sampled model output or an agentic serving
completion.

Every execution emitted 1,320 ordered pre/post semantic records. All twelve
paired full record vectors and output hashes matched. Immutable source checks
required both selected full 248,320-value logit rows and 5,120-value nextn
hidden rows to be finite and bit-identical. The retained independent evidence
is digest/shape metadata rather than a separate full-float payload oracle.
The persistent allocation census remained 260 distinct roots; no speculative
capture roots were included. No MTP tail, sampler update or cost-controller
update ran.

Each shared sample recorded exactly 288 chosen checked events: 64 FFN
anchors, 96 retained Delta completions and 128 retained row-copy completions.
The event order was FFN anchor first for each layer, followed by retained
submission order. Every recorded own command-buffer result was healthy.
The FFN host interval includes queued upstream work; retained row/Delta
intervals measure only their post-anchor waits. Row submission cost remains
inside the whole-target timer. Chosen command-buffer durations do not cover
the whole graph and their sum is not GPU time union or pure FFN attribution.

The parent completed with child exit zero in 389.7010763329454 seconds,
761 runtime observations, one actual Rust test passed and verified empty
owned process group after cleanup. Maximum observed sample-start gap was
525.026458 ms and complete observation duration 53.765416 ms. Pressure stayed
normal, swap-used growth was zero, and thermal observations were 24 Nominal
and 53 Fair; this was not a nominal-only performance qualification. Process
RSS/footprint measurements were raw process ledgers, not complete Metal or
unique physical host ownership. Full retained historical metadata digests
matched before/after and new mutable evidence stayed within its reserved
I/O allowance. No physical admission or release credit follows.

## Disposition

Keep this spike and evidence as a failed speed hypothesis. Correctness of
this bounded producer case does not qualify complete S3, F2, Q1, full AUTO,
width-three/four execution, production serving performance or physical fit.

This run does not establish why the shared arm was slower. Later proposals
may inspect remaining forced helper fences or different bounded group
boundaries, but must undergo separate source review and prospective controls.
Do not subtract hypothetical savings from this result or compare separate
runs as a causal intervention. Existing earlier failed/incomplete trials
remain failed/incomplete and are not rescored by this record.
