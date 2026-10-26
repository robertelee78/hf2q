# Gemma full-input checkpoint: four-agent serving proof

Status: the default-off experimental mechanism passed the fixed four-agent
coding and seven-wire replay controls. Default promotion, whole Q5 acceptance,
physical-memory admission and performance wins remain unproved.

## Executed identity

The clean server commit was
`18162bb702806ac6c083bf01ec80f4675207fda4`, binary SHA256
`ff655efd5013a11ad608330f35b3622c2e448f019dcc4e94c8c483eb03cd09e1`,
using public `mlx-native =0.15.5`. The exact Google Gemma GGUF SHA256 was
`bf65b7e163216885790442d0324839101ba4b5b49f7be4ec0cf3d7c5c538ef53`.
Retained artifact identity combines prior full-byte verification with current
device/inode/size/mtime continuity; this run did not rehash the model payload.
The host was Apple M5 Max, 128 GiB, macOS27.2 build26B5112a.

The profile kept four SlotAware slots, logical context262144 per slot,
Hybrid/TQ8, no graft/projector, repetition penalty1.05, output limit8192,
LCP8GiB and KV48GiB. Both `HF2Q_GEMMA_FULL_INPUT_CHECKPOINT=1` and
`HF2Q_UNSAFE_EXPERIMENTS=1` were explicit. The startup summary confirmed the
selected experiment and its1536MiB requested CPU-anchor quota. The original
GGUF was exposed through an owned text-only hardlink locator, not copied or
modified; cleanup removed only that locator and restored original link count.

The fixed client uses the checked-in four-agent coding protocol and explicit
Gemma adapter. The retained client source was frozen at
`fec19fae` before execution. The exact original requests, responses, source
files, approvals, tool logs and receipts remain in the private campaign packet
`gemma4-full-input-n4-20261026T210724Z`; no generated data is checked in.

## Actual behavior

All28 requests completed with HTTP200, including12 SSE responses with one
finish/usage/DONE each. The24 actual tools produced four correct checked
`u64`-parse/range-check/`u16` patches. Each agent's ports2 and all4 Rust test
results were independently verified from raw tool records. Tool-result
continuation and the exact message-history chain passed. The server PID54847
and client PID54961 exited0, were reaped, and left empty owned process groups.

The counts below held for all four agents; their token digests were distinct.

| Turn | Actual prompt tokens | Cached tokens | Prior accepted anchor |
| --- | ---: | ---: | ---: |
|0|465|0|none|
|1|514|458|458 strict cue-less anchor|
|2|714|514|514 full input|
|3|943|714|714 full input|
|4|1249|943|943 full input|
|5|1408|1249|1249 full input|
|6|1788|1408|1408 full input|

After verified unload, the separate engine-free family helper ran once for
each agent. All four exact seven-wire replays passed; no model was loaded
during replay. The helper binary SHA256 was
`52e1e5212b402b54de13c7c358e089aa441ba494234cb2c04f076652cc420a74`,
from committed source `9b671328`. Its engine and checkpoint source bytes match
the executed server. The generic strict oracle remains unchanged.

Each replay bound the production-normalized request/render/tokenizer, raw
unary/SSE response usage and completed HTTP exchange to the actual capture
length, little-endian token digest and slot. All24 capture identities were
unique. Agents0/1/2/3 used slots0/3/1/2 consistently. Every growing edge
required the entire prior accepted anchor, with no minus-one allowance.
Output status was `GemmaFullInputCheckpointSevenWireQualified`; generic strict
oracle and model-wide qualification remain false.

Independent review verified all1196 original packet hashes, all19 replay
packet hashes, their bound inputs, source/build identity, four patches, raw
usage joins and cleanup. This establishes the mechanism only for the observed
465–1788-token conversations, not largest-context stress or cancellation.

## Operating budget and retained failures

The preceding `gemma4-full-input-n4-20261026T205547Z` attempt held before load: its
109.063GiB floor exceeded107.32GiB available working memory. It stays held.
The next prospectively reviewed packet corrected one scratch-count term:
the default tiled live loop tiles sliding layers but consumes the complete
remaining query in each global layer. Thus four lanes times five global
layers require at most20 dense expansion pairs per call, rather than the
previous35 sliding-ragged overbound. Each pair retained the admitted-input
bound253952 from full262144 context minus8192 output reserve.

This saves15602810880 requested bytes while retaining all other broad cells
and the prior20.125GiB operating allowances. The revised explained operating
amount is101502938736B; an extra1GiB tool-process allowance still fits below the
prospective96GiB floor, leaving502534544B additional padding. Ten fresh baseline
samples preceded the successful load. Free+cached-files is an operating metric,
not qualified physical cache credit. Runtime pressure/memory/process-ledger
guards stayed active. No physical-partition or formal fit claim follows.

The earlier rejected V7 argument excluded possible installed live lanes and
remains rejected. This correction retains all four lanes. The earlier V2
primitive comparison failure between different prefill histories also remains
failed; the governing ADR records its0.812623 maximum logit discrepancy and
the subsequent same-image exact-bit control. No failed packet was rescored.

## Remaining gates

A matched repeated speed study is still required before any speed claim.
The broader unit run recorded5402passes,5failures and78ignored cases.
Serial-mode source checks were repaired and23 focused controls passed.
Separate fixes address stale native-version evidence and DeepSeek delimiter
grammar resource expansion. The whole regression suite is not yet green.
Default promotion, cancellation/overshoot, largest stress and all-four model
performance acceptance remain separate gates.
