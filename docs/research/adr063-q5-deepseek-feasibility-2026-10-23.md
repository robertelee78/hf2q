# ADR-063 Q5: DeepSeek full-capacity feasibility finding

Status: independently reviewed source arithmetic and device-property observation.
No model was loaded. This is a rejection under the proposed conservative policy,
not proof of a hard Metal allocation limit or permission to change that policy.

Source basis: hf2q `4cd1e79c49a836f2eb8f2ccdea5d87bc256d2a20`, native
`f92eb020c4d3f821700e648fbce15d6cf75c2cc6` / published 0.15.1. The subsequent
F1 merge changes no cache-allocation formula used here. Frozen DeepSeek artifact
SHA256 is `50df7e3dfe06321613c78075df4111b14985a0db31087992d7438f2895a3aad6`.
See the [memory design](adr063-q5-memory-design-2026-10-23.md) and the separate
[incomplete process-metric coverage finding](adr063-q5-metric-availability-2026-10-23.md).

## Known resource terms

Each of four slots has 262144 token positions. The loaded model separately owns
an eager cache at 131072 positions; current source retains that boot cache
(`engine_deepseek4.rs`, `load_with_context`, session provisioning and
`ensure_cache_capacity`). Do not divide context by four or omit the boot owner.

The artifact declares 43 layers: two compression-ratio 0, 21 ratio 4, 20 ratio
128; attention head dimension 512, index dimension 128 and window 128.
`Deepseek4CachePlan::for_context` and its allocation in `deepseek4/cache.rs`,
with `cache_buffers.rs`, give deduplicated live root bytes:

```
linear_bytes_per_token = 21*(512*2/4 + 128*2/4) + 20*(512*2/128) = 6880
fixed_live_bytes = 43*128*512*2
                 + 21*(2*8*1024*4 + 2*8*256*4)
                 + 20*(2*128*512*4) = 17842176
A(context) = 6880*context + 17842176
A(131072) = 919617536
A(262144) = 1821392896
four_full_plus_boot = 4*A(262144) + A(131072) = 8205189120
```

Window and compressed views share an attention parent and are counted once.
These are resource capacities, not a claim that every lazy page is physically
resident at every instant.

The shared 8 GiB scheduler setting is an accounting/admission budget, not an
extra 8 GiB allocation, not 8 GiB per slot and not a whole-process peak bound.
Four full scheduler charges are `4*(262144*6880+50331648)=7415529472` bytes
(6.90625 GiB). That inclusive fixed allowance cannot be added again to the
actual cache roots and anchor roles.

The directory's 1328 nonoverlapping encoded payloads total 107426004892 bytes.
Independent catalog reconstruction accounts for exactly the same names:
`6 + 43*24 + 41*4 + 21*6 = 1328`, with no missing or extra tensors. This is not
a sum over unused extra weights. `deepseek4/residency.rs` retains every required
`ResidentTensor` as an `MlxBuffer`: raw/I32 tensors keep mapped roots or device
buffers; elementwise F32 expansion cannot shrink the encoded types present.
Native `GgufFile::map_tensor_data` creates shared Metal resources, and tensor
views retain their root storage. Consequently encoded payload sum is a valid
lower bound on unique weight resource capacity. Mapping padding, expanded
copies and other roots can increase it.

## Two separate inequalities

GiB means 2^30 bytes. Against the host/process cap, known weights plus the five
live caches total 115631194012 bytes (107.689941 GiB), leaving 8.310059 GiB
below 116 GiB **before competing memory and all omitted owned terms**.
An explicitly conservative envelope adding four committed and four pending
compact anchors plus one old cache during growth totals 108.679336 GiB,
leaving 7.320664 GiB before those omissions. That overlap envelope is not a
proved runtime upper bound: simultaneous roles and growth serialization require
their own proof. Neither positive residual proves host admission.

A separate no-model Metal device query reports:

- Apple M5 Max, unified memory true;
- `recommendedMaxWorkingSetSize = 115448725504` bytes (107.520004 GiB);
- `currentAllocatedSize = 65536` bytes during device initialization only.

The query source creates the device and reads properties; it submits no GPU
work and loads no model. Its structured receipt binds source hash, argv,
timestamps, exit 0 and output: 987 bytes, SHA256
`2b278618c8209519eca81e8a000c386e53f3a3db50298055041c2bf28c231f57`.
The installed SDK's `MTLDevice.h` describes this recommendation as an
approximation for good performance, with likely overcommit/performance penalty
above it. It does not define a hard allocation maximum. Observed zero-valued
`iogpu.wired_limit_mb` and `iogpu.wired_lwm_mb` have not been established as an
explicit limit or unlimited allowance.

Under the design's proposed `G <= W` policy, charging unique full resource
lengths and using that recommendation as W:

```
G_lower = 107426004892 + 8205189120 = 115631194012
W = 115448725504
G_lower - W = 182468508 bytes = 174.015530 MiB > 0
```

The frozen full-capacity current implementation therefore fails that policy
before anchors, scratch, mapping padding, logits or queued resources. CPU-only
categories were not needed for this inequality. This is conditional on the
declared full-capacity shape, not an instantaneous physical-footprint claim.

Complete scratch/ownership bounds, process-metric coverage and host admission
remain open. A different device policy needs separate reviewed evidence before
execution. Raising W after a failed run, interpreting sysctl zero as infinity,
or reducing the matched context/task cannot manufacture acceptance. Removing
unneeded boot ownership or reducing resource lifetimes without reducing the
workload would instead require its own implementation design and proof; neither
is selected or approved here. The pure ledger may demonstrate the rejection,
but cannot make this current full shape admissible.
