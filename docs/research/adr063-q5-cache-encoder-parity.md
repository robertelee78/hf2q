# Q5 synthetic fresh64 cache-write control

This is test infrastructure, not a warmup capacity/policy change. Production
warmup remains unchanged. It uses the exact published registry mlx-native0.15.2
pin and no local allocator API. No model, weights, EOS embeddings, hidden/logits,
head, attention/indexer selection or full43-layer execution is tested.

Private production extractions preserve layer allocation order/state reset,
postprojection main/indexer compressor tails and the compressed window-write
segment. Original callers still execute learned projections and surrounding
barriers/scoring in their original order. The ignored control calls those same
tails for actual ratio4/128 LayerCachePlans at128 and131072 capacity. Ratio0 is
explicitly allocation/layout plus the native copy primitive only; the compressed
production encoder does not accept ratio0. No partial full cache/cursor is forged.

The fixed official plan fixture retains43layers, original context65536/max1048576,
window128, head512/index128/rope64 and compression schedule. Fresh0/64rows/graft0
has16 completed ratio4 groups and zero ratio128 groups. Native compressor still
receives each capacity's actual cache_len and performs state updates for ratio128.

## Explicit requested packet

Temporary roots, each constructed once and reused only after completion:

| Root | Dtype/maximum shape | Bytes |
| --- | --- | ---: |
| Main projected KV | F32[1,64,1024] |262144 |
| Main projected scores | F32[1,64,1024] |262144 |
| Main output | BF16[1,16,512] |16384 |
| Main rope/quant output | BF16[1,16,1,512] |16384 |
| Main APE | F32[128,512] |262144 |
| Main norm | F32[512] |2048 |
| Indexer projected KV | F32[1,64,256] |65536 |
| Indexer projected scores | F32[1,64,256] |65536 |
| Indexer output | BF16[1,16,128] |4096 |
| Indexer rope/quant output | BF16[1,16,1,128] |4096 |
| Indexer APE | F32[4,256] |4096 |
| Indexer norm | F32[128] |512 |
| Post-RoPE window source | BF16[64,512] |65536 |
| Compressed positions | U32[64] |256 |
| YaRN frequencies | F32[32] |128 |
| **15 temporary roots** | |**1031040** |

Actual representative layer roots across both capacities total20roots and
45,032,448bytes: paired ratio0 262,144; ratio4 42,409,984; ratio128 2,360,320.
Attention parent owns window/compressed zero-copy views; these are not additional
roots. All requested Metal payload across the control is46,063,488bytes, counting
historical requests rather than inferring release from Drop. Registry/pipeline,
GPU threadgroup, command/residency/driver/allocator metadata is separately Unknown.
Requested payload is not physical footprint or host/GPU admission.

All physical logical bytes are initialized before reads. The HF pure YaRN builder
runs once with unchanged configuration; the distinct native keyed RoPE cache is
not used. Smaller shapes are dense prefix views of these roots. Native parameters
are inline bytes, not additional Metal buffers. The full existing learned-matmul/
query/attention arenas are deliberately absent.

Host comparison retains one1MiB reference arena and no full-large-cache copy.
Nested public plan/span/cfg Vec capacities, packet argument shape capacities and
a finite local-view allowance must fit1MiB bookkeeping;64KiB optional chunk,
256byte frequency allowance and1024byte inline allowance bound the scoped packet
at2,163,968bytes. Opaque native/Arc/framework overhead is excluded and Unknown;
this is not total process host memory<=4MiB. No accumulating per-case snapshots.

## Oracles and lifetime

Both capacity arms are fully encoded before either commits on the same queue;
both completion results are collected before assertions/readback. Cache roots,
packet roots and registry remain alive. No CPU payload accesses occur between
submissions. Failed encoding before either submission publishes no cursor;
external process-group watchdog/kill recovery is required for stalled drains.

The independent window oracle computes exact BF16 small signed integers from
(row,col), separately from both GPU results. Tail canaries are checked against
each arm's original finite value. A test-only omission at the shared production
window seam makes both windows retain the SAME canary: their equality passes,
but both independent expected-write oracles must reject. Compressed destination
rows additionally match their drained rope/quant source, with every row differing
from its initial canary somewhere. This detects copy omission, not common-mode
compressor arithmetic error. State raw bits compare across arms. Ratio128 has no
completed destination rows and must leave compressed storage untouched.

A separate actual native cache_len15/16-group rejection is never submitted.
Production state reset checks KV0/score-negative-infinity without claiming append
storage was erased. No completed full-cache cursor or recovery-owner proof follows.

## Execution boundary

The integration owner built and executed the controls described below; the
source/control author and independent reviewers did not rerun them. Pure packet
arithmetic test:

```sh
cargo test --locked --bin hf2q explicit_packet_has_fifteen_roots_and_exact_requested_payload -- --nocapture
```

The ignored control requires independent source review, explicit authorization,
a suitable Apple Silicon host, reviewed memory preflight and external process-group
watchdog before execution:

```sh
cargo test --locked --bin hf2q actual_cache_write_capacity_parity_fresh64 -- --ignored --nocapture --test-threads=1
```

Profiling/trace/auto-barrier overrides, programmatic capture and unretained-reference
environment switches listed in the test must be absent. No model or downloads.
Zero exit alone is not proof: require six Q5_CACHE_ARM completed records, every
Q5_CACHE_COMMANDS small_ok/large_ok true (including the common omission control),
both negative markers, host packet record and final Q5_CACHE_CAPABILITY completed
with exact15/20root and46,063,488byte counts. unavailable_or_unknown is a separate
nonqualification outcome. This control cannot justify warmup capacity adoption,
64EOS output parity, speed, complete memory bounds or fullQ5 qualification.

## Source-bound execution — 2026-10-24

The bounded synthetic control passed once on Apple M5 with the published
registry mlx-native0.15.2 (VCS `7bb30a2fac783e0b340ec2f2086289f9751dda7a`,
crate checksum `e09c782f1b774876674d04c15f03fae2f81b58dfe6401aba87153dcbe9d3a3a4`).
The exact fully qualified ignored filter matched one test: one passed, zero
failed, zero ignored. Supervisor exit0, no timeout, elapsed254,815,792ns. This
execution duration is not a performance comparison.

Independent postexecution review rehashed all13 frozen files, the retained
executable and log. The compile freeze differs from the original source freeze
only by removal of an unused DType import; the preceding strict compile failure
ran zero tests and is not counted as validation. The pure packet arithmetic test
subsequently ran exactly one test and passed.

All six completed arm records appeared: ratios0/4/128 at capacities128/131072.
Ratio0 remained allocation/layout/copy-primitive scope. Ratios4/128 exercised
the production postprojection tails. Both capacities reported matching semantic
lengths:65,536bytes for ratio0,167,936 for ratio4 and589,824 for ratio128.
All four command-pair records (three positive pairs plus the common-omission
pair) reported both small_ok=true and large_ok=true. Both negative markers ran:
short_cache_rejected=true and common_omission_rejected=true with equal broken
windows. No unavailable_or_unknown marker or skipped capability occurred.

The host marker reported99,276bytes of scoped bookkeeping and1,048,576bytes of
reference storage. Adding the declared65,536-byte optional chunk,256-byte
frequency and1024-byte inline allowances gives1,214,668bytes, within the declared
2,163,968-byte scoped bound. This is not total host footprint. Final capability
reported15temporary roots,20cache roots and46,063,488 requested payload bytes,
with model_loaded=false, runtime_admission=false and opaque_metadata_known=false.
Historical requested bytes are not a measured physical peak or proof of final
Metal resource release.

| Evidence | SHA256 |
| --- | --- |
| Compile/execution source freeze | `00ae5fdc6c10fd402e0c7d8a23d254df874ac5f2a1faf0a34e012aeefa304b62` |
| Runtime test executable | `40cb266901a2a5d2083636bf3481ed596469ce0e87ba214d82851673a9b77897` |
| Control log | `9de92f7d645a2a45a49813c6a7d6feebe87f9f81984d6f52140a1aaff97334b6` |
| Executed encoder_parity.rs | `2420c5ea25829dca6ddd8360ec3330ec52be3c6145c8384c52475544b5f90341` |

Raw evidence, full source archive and executable remain outside Git. The
pre-execution freeze retains executed=false and its original documentation
hash; the separate execution receipt establishes the result. This results-only
documentation update does not alter tested Rust source. The external supervisor
checked source and binary identities before/after and covered the process group
with a180second deadline. Its meminfo preflight was context for this small
model-free control, not qualification of a model admission observer.

The PASS is fresh64-row synthetic postprojection cache-write/state-tail capacity
parity with the documented independent window/copy checks. Common-mode compressor
arithmetic outside those checks remains possible. No64EOS/model output parity,
learned projection/attention correctness, full warmup, capacity-policy adoption,
speed, complete memory bound or fullQ5 admission follows. The separately required
selected-path memory inventory and representative model validation remain open.
