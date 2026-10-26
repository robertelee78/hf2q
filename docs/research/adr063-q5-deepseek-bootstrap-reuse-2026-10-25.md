# DeepSeek bootstrap cache reuse

Source candidate originally based on f08a4f409cbe23502e8a002effaa480d612d3bf2,
validated with the verified native0.15.4 dependency pin at14e2bb97. Independent
source review, root compilation and tiny ownership controls passed. The source
author performed no Cargo, test, model or device execution. The original draft required separate full-model validation; the scoped
public0.15.5 real-model results are recorded below.

Eligible SlotAware startup now stages one ordinary eager128 cache and n−1 normal
lazy initial sessions, then swaps the original pristine loaded bootstrap into
slot0. All resources, metadata reservation and validation precede that single
cache swap and publication. The shared empty-session constructor moves its owned
cache; it does not clone tensor payload. A reserve, allocation, initialization or
validation error leaves the existing execution surface and publication unchanged.
There is no throwaway normal-sized slot0 allocation.

The optimization requires an ungrafted pristine initial bundle and no published
sessions. Every token/logit/anchor/transaction field is checked explicitly, with
position, poison, capacity and actual CachePlan metadata. Otherwise ordinary
provisioning remains selected. Repeated provisioning retains its existing
replacement semantics; it is not a new no-allocation idempotence promise.
FIFO, warmup, cancellation, growth, restore, persistence and worker swap brackets
remain unchanged. No new controller, role system, toggle or native API is added.

For the selected43-layer geometry, actual CachePlan descriptors give
A131072=919617536B, A262144=1821392896B and A128=18722816B. Four full slots plus
smaller bootstrap require7304294400B rather than8205189120B, saving900894720B
requested payload (~0.839GiB). Add919617536B separately for one old source cache
during growth. This is source requested-capacity arithmetic, not a measured
physical release or complete B/C/R/G/admission bound.

The old bootstrap is eager; the other original sessions are lazy. Agent0 therefore
inherits eager residency hints while other sessions retain their lazy allocation.
The large eager root already existed; it is moved, not newly materialized.
Numerical plans/valid-row bounds remain compatible, but equal per-agent first-touch
or residency behavior is not claimed. The replacement ordinary128 eager cache
supports the unchanged supervised64-EOS warmup, repeated/late warmup and reset.
“Sentinel never executes” was a discarded design constraint, not a product rule.

Six proposed pure tests use production staging/commit/planning seams: selected
actual descriptor arithmetic; one/four independent transfers; every pristine
field and fallback; reservation/allocation/validation failures before commit;
context/repeated fallback; anomalous reservation before cache creation. The pure
shape fixture is not a full model or runtime serving proof.

One ignored actual CPU-accessible Metal control is source only, pending separate
packet review and root execution. It uses the same production staging/transfer
helpers with actual three-layer ratios0/4/128 caches, head32/index16. Cases1and4
start at capacity256, replacement128, grow active slot0 to512. All physical roots
are initialized by index-sensitive CPU bytes. It verifies independent root
identities, two swap directions, snapshot rollback and migrated restore, every
valid logical section and bootstrap/other-slot isolation. No command buffer,
encoder, GPU dispatch, model/tokenizer load or serialization image is used.

| Requested payload role | One-slot case | Four-slot case |
|---|---:|---:|
| Original cache256 |69,760|69,760|
| Eager replacement128 |66,624|66,624|
| Additional lazy caches256 |0|209,280|
| One compact snapshot |63,488|63,488|
| One growth destination512 |76,032|76,032|
| Conservative case total |275,904|485,184|

Both cases run sequentially in one process: historical requested Metal payload
761088B, planned108 roots (39+69), maximum simultaneously source-owned requested
payload at most485184B. Keep the historical sum as a conservative resource-payload
envelope if earlier native resources remain retained. The control reserves a
separate16KiB numeric host bound for small identity collections; plan/shape/driver
metadata and runtime initialization remain explicitly separate. No full-payload
CPU readback arrays or images are created. Temporary MlxBuffer clones only retain
existing allocations and are dropped after CPU writes. Snapshot roots are true
copies and are included. These counts derive actual plans and source paths;
they are not global allocation-counter observations.

Root proposed filters (not executed by author):

```sh
cargo test --locked --bin hf2q ds_bootstrap_ -- --test-threads=1
```

The actual ignored filter is
`serve::api::engine_deepseek4::slot_provision::tests::ds_bootstrap_actual_cpu_transfer_restore_growth`.
Require one passed test plus both case markers and exact final
DS_BOOTSTRAP_CPU_COMPLETE marker; a Metal initialization failure panics, not skips.
Actual execution requires a separately reviewed supervisor and frozen binary.
Real supervised warmup, four-slot tool/prefix/cancel/growth quality and all-four
speed qualification remain later obligations. No Q5 fit or model speed win is
claimed by this allocation change.

## Retained root validation

Locked all-target/all-feature check passed in27.39s, release build in35.60s
and test-binary compilation in49.99s against published registry native0.15.4.
The actual ds_bootstrap_ filter ran6pure tests successfully and left1hardware
control ignored. Six separate supervisor classification controls passed. One
reviewed fresh-process CPU-accessible Metal control then passed both one-slot
and four-slot cases, all three full markers and exact1passed/0failed/0ignored
result in0.08s. The source-derived historical requested envelope is761088B;
maximum source-owned live payload485184B, with no GPU commands or model weights.
Independent retained-evidence review accepted this narrow result. Those initial checks alone did not prove full-model serving, physical fit or
speed. The following real-model evidence completes the scoped serving gates;
whole-Q5 memory qualification and all-family performance remain separate.


## Public0.15.5 real-model validation

The executed clean source was2d7599907be92d17736bc205806907589e845306,
binary SHA256111649ce1b0d635d0a1403f34004667e1ab61757ff203c68c513dcac912488c2.
Cargo pinned public registry mlx-native0.15.5, archive checksum
be6b17fcc030f03615f66873ba22e59aa29b5ae9ff30d6370cbdedf8649245e9.
The model was DeepSeek-V4-Flash-0731-hf2q-deepseek4-agentic-q2.gguf,
107431343264B, SHA256
50df7e3dfe06321613c78075df4111b14985a0db31087992d7438f2895a3aad6.
Hardware: Apple M5 Max128GiB, macOS27.2 build26B5112a. Each packet ran one
full-model process with262144 logical tokens per slot and four slots; every
owned server/client group was reaped and empty before the next load.

The combined candidate includes the separately landed DSML delimiter correction;
that correctness fix is not attributed to bootstrap reuse. Independent source
comparison binds the bootstrap/cache implementation in this PR to the actual
executed candidate. Later main changes to diagnostics and the unwired declared
ledger do not turn a newer source commit into an executed binary identity.

- `deepseek390-n4-20261026T155503Z`:28 real assistant requests,24 actual tools,
  four correct patches, and twelve raw Rust test outputs with two passing tests
  each. Unary/SSE framing, one-call semantics, actual tool-result continuation,
  cleanup and all four exact native eight-token-tail prefix replays passed
  independent review. The prefix helper's configuration checks are composed
  with actual launch/source evidence; they do not independently attest a binary.
- The canonical cancellation child in
  `deepseek390-cancel-growth-20261026T164811Z` passed independently: three
  cancellations/recoveries, cached-token counts17490/17490/60607, stable counter
  progression0→1→2→3, semantic peer output endingPEER_DONE, bounded transaction
  overlap and verified cleanup. Its parent remains FAILED because its later
  growth stage exhausted a32-token response allowance. No parent was rescored.
- `deepseek390-growth-only-20261026T174508Z`:the repaired standalone growth gate
  passed. It processed125513 prompt tokens, migrated125640 tokens from capacity
  131072 to262144 in95.039ms, then reused125505/125619 tokens with1057.916ms
  semantic TTFT. The final request reused125652/125672 tokens:exactly20 suffix
  tokens, terminal sentinel, stop finish, and verified cleanup. Independent
  review rehashed11256 retained artifacts with no mismatch.

The growth harness preserves the actual assistant message including reasoning;
content-only reconstruction had forced an older recovery checkpoint in the
preceding failed packet. Its required-tool/continuation/short-tail allowances
were1024/8192/1024. The cancellation harness used required-tool1024 and peer8192,
with its original continuation/cancellation limits unchanged. These are genuine
native requests, not scripted assistant output. Reproduction uses
`scripts/test_deepseek4_cached_suffix.sh`,
`scripts/test_deepseek4_long_context_cache.sh`, and the explicit DeepSeek client
contract in `adr063-deepseek-coding-diagnostic.md`; freeze exact source, model,
settings, monitoring policy and tool sandbox before executing.

The other failed packets remain failed evidence. These separate successful
scenarios establish scoped warmup, independent multi-turn slots, prefix reuse,
cancellation/recovery and lossless growth. They do not establish simultaneous
largest-context four-slot stress, a disjoint physical-memory upper, measured RSS
saving, peer parity, an all-four speed win or completion of Q5. The900894720B
saving above remains source-derived requested capacity, not measured physical
release. No performance claim is made by this PR.
