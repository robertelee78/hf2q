# Hugging Face download performance RCA and native-Xet decision

- **Date:** 2026-08-26
- **Scope:** hosted GGUF and native safetensors retrieval in `src/input/hf_download.rs`
- **Observed report:** a 15 GiB artifact remained below 50% after 59 minutes in
  hf2q while another Hugging Face client completed the artifact in under three
  minutes.
- **Decision:** replace `hf-hub 0.5`'s synchronous transfer path with pinned
  `hf-hub 1.0.0` and its native Xet integration; retain hf2q's stronger
  exact-commit, exact-selection, and post-download digest gates.

## Executive conclusion

The primary cause was client selection, not hashing or quantization. hf2q put
every large object through `hf-hub 0.5`'s synchronous `ureq` implementation.
That implementation downloaded one file through one HTTP response stream, and
hf2q called it from serial loops. The hosted-GGUF path was the worst case: one
15 GiB file meant one transfer stream with no file-level concurrency available.

The same old crate already contained an unused async ranged mode whose `high()`
preset documented a goal of saturating links above 500 MB/s. hf2q never selected
that API. More importantly, the Hugging Face Hub has since moved large-file
storage to Xet. An old client is served through the compatibility LFS/Xet bridge
and receives one signed object URL; an Xet-aware client obtains reconstruction
metadata and concurrently fetches content ranges from CAS storage.

The user's timing bounds are consistent with this source-level result:

- less than 7.5 GiB in 59 minutes is less than 2.17 MiB/s;
- 15 GiB in 3 minutes is about 85.33 MiB/s; and
- the observed clients therefore differed by more than 39x.

Those figures localize a transfer-path problem, but they are not a controlled
benchmark: model, cache state, route, machine, and exact competing client build
were not captured. The serial legacy path is proven; attribution of every bit
of the 39x difference to Xet alone is not.

The comparison to Unsloth is directionally confirmed. Current Unsloth depends
on `huggingface_hub`, and its checked-in CI downloader explicitly states that
current Hub transfers route through `hf-xet`; it enables Xet high-performance
mode and 64 parallel range gets. That proves the competing ecosystem actively
uses the transport class hf2q lacked. It does not prove the anecdotal Unsloth
application run used the same flags, cache state, or route. The same script also
has a stall watchdog because Unsloth observed a per-file Xet hang, so hf2q must
treat cancellation/stall recovery as a reliability follow-up rather than assume
Xet is infallible.

## Before: exact call graph and bottlenecks

### Hosted GGUF

```text
resolve_hub_gguf_catalog
  -> resolve repository and immutable commit
  -> HEAD each candidate, serially

download_hub_gguf / download_hub_companion
  -> HEAD the selected file again
  -> hf_hub 0.5 ApiRepo::get(filename)
       -> cache lookup
       -> HEAD metadata again inside hf-hub
       -> one ureq GET / one response body / std::io::copy
  -> read the completed file again for hf2q SHA-256 verification
```

A single hosted GGUF has no shard-level parallelism. The old synchronous client
did support continuation from its `.part` length with one `Range` request, but
it did not split a fresh object into concurrent ranges.

### Native safetensors conversion

```text
download_model_reference
  -> resolve_native_source_plan
       -> repository info / immutable commit
       -> serial metadata HEADs
       -> serial small-metadata downloads and verification
       -> authenticate index
       -> serial weight-shard HEADs
  -> download_via_hf_hub
       -> resolve repository a second time
       -> serial initial-file HEAD + get + verify
       -> for every selected weight shard, serially:
            HEAD + get + verify
```

This path exposed both file serialization and duplicated repository/metadata
work. Multiple shards could never be in flight together. Each large shard also
used only one fresh HTTP stream.

## Ranked root causes

| Rank | Cause | Confidence | Effect |
|---:|---|---|---|
| 1 | hf2q selected `hf-hub 0.5`'s synchronous single-stream downloader | Proven from pinned source and hf2q call sites | Dominates a single 15 GiB GGUF and limits every shard independently |
| 2 | hf2q serialized selected files | Proven from both download loops | Prevented aggregate bandwidth across safetensors shards |
| 3 | the old client used the Hub's compatibility LFS/Xet bridge, not native Xet reconstruction | Proven from official Hub migration design and response behavior | Missed concurrent CAS range retrieval and Xet deduplication |
| 4 | native conversion resolved the repository twice and repeated metadata requests | Proven from call graph | Adds latency, especially for many shards; not enough to explain an hour by itself |
| 5 | hf2q performs a full local SHA-256 pass after download | Proven | Required trust cost; tens of seconds at ordinary local SSD rates, not a plausible 39x network gap |
| 6 | route, CDN, proxy, Wi-Fi, throttling, or competing cache state | Possible, unmeasured | May explain some of the anecdotal delta but is not required to reproduce the architectural bottleneck |

The former module documentation said `hf-hub 0.5` restarted an interrupted file
from byte zero. Inspection showed that statement was stale: the synchronous
client retained a partial file and sent a range request from its current length.
That was useful restart behavior, but it remained one sequential range rather
than a concurrent fresh-download strategy.

## SOTA evaluation

### Chosen: `hf-hub 1.x` native Xet

This is the official in-process Rust path and preserves the product boundary:
downloads remain Rust-native while hf2q continues to own conversion,
quantization, and inference. `hf-hub 1.0` detects `X-Xet-Hash`, creates a Xet
session, obtains CAS reconstruction information, downloads ranges concurrently,
and publishes the standard Hugging Face cache blob/snapshot layout only after
the transfer succeeds.

For bounded snapshots, the library also resolves the revision once, filters the
tree, fans metadata requests out concurrently, batches Xet objects, and runs up
to eight file downloads concurrently. hf2q now supplies only literal, escaped
allow-patterns for its already-selected weight list. Every selected
`.safetensors` or `.gguf` must advertise Xet; there is no silent large-payload
downgrade to the legacy HTTP stream.

### Not chosen: deprecated `hf_transfer`

`HF_HUB_ENABLE_HF_TRANSFER` is deprecated in current Hugging Face tooling and
`hf_transfer` is no longer the preferred transport. The official replacement is
`hf-xet`. Adding a legacy environment toggle would optimize an obsolete path
and still leave hf2q on an old client API.

### Not chosen: Python, `hf` CLI, Unsloth, aria2, or curl subprocesses

These can be useful operator benchmarks, but production subprocess retrieval
would add a second dependency/runtime/cache/error model and violate hf2q's
in-process Rust download boundary. The native Rust Xet client supplies the same
class of concurrent range reconstruction without outsourcing product behavior.

### Qualified-host amendment: `HF_XET_HIGH_PERFORMANCE=1`

Normal Xet already uses adaptive concurrent range retrieval. Hugging Face's
current Xet documentation says the adaptive defaults are tuned to saturate most
network paths and reserves high-performance mode for high-bandwidth machines
with at least 64 GB of RAM. Xet 1.5.3's source raises its download-buffer limit
from 8 GB to 64 GB in that mode. Forcing it would therefore be unsafe on part
of hf2q's Apple-Silicon support range. The 2026-08-27 exact-artifact
interleaved A/B measured a 251.23-second adaptive median against a 225.99-second
high-performance median (10.05% lower wall time), while median peak RSS
increased from 3,301,113,856 to 10,223,222,784 bytes. hf2q therefore enables
the upstream preset only on Apple Silicon with at least 64 GiB physical memory
and only when neither upstream mode variable is explicit. Adaptive remains the
default everywhere else. This is one production transport with a resource
qualification, not multiple download implementations.

### Fallback considered: old async `hf-hub 0.5` high mode

This would have removed the immediate single-stream bottleneck, but it would
retain the legacy bridge and an obsolete client/cache implementation. It is a
useful confirmation of the RCA, not the landed architecture.

## Landed transfer contract

1. Resolve a mutable reference to one exact 40-hex commit before payload
   transfer.
2. Bound and validate the repository inventory and selected paths.
3. Fetch exact-origin metadata without following the absolute CDN redirect, so
   `X-Repo-Commit`, `X-Linked-Etag`, `X-Linked-Size`, and `X-Xet-Hash` remain
   available to hf2q. This works around a `hf-hub 1.0` public metadata-helper
   defect found by the live test; its payload downloader uses a correct internal
   no-absolute-redirect path.
4. Require Git SHA-1 for selected metadata and LFS SHA-256 for safetensors/GGUF
   payloads.
5. For one hosted artifact, call `download_file` at the exact commit. Xet is
   selected automatically when the Hub advertises an Xet hash.
6. For native conversion, reuse the existing immutable source plan and call one
   `snapshot_download` for exactly the selected weight filenames, with eight
   file workers. Literal glob escaping prevents metacharacters from expanding
   authority.
7. Require every selected output to appear under the exact commit snapshot.
8. Re-read and verify every file with hf2q's SHA-1/SHA-256 verifier before
   conversion, publication, or serving authority is granted.

Transport never becomes trust authority. An Xet success without hf2q's local
digest match is a failure.

## Cache, interruption, and progress semantics

- Cache layout stays compatible with standard Hugging Face
  `models--OWNER--REPO/blobs` and `snapshots/<commit>` paths, so existing exact
  snapshots remain reusable.
- The library writes `.incomplete` objects and renames them into cache blobs
  only after successful HTTP/Xet completion. Completed blobs survive retries.
- hf2q does not promise that an interrupted Xet reconstruction resumes from the
  exact output-file byte offset across processes. Xet may reuse CAS/chunk-cache
  state, while the final incomplete object remains unpublished. The safety
  guarantee is atomic publication plus completed-blob reuse.
- Native Xet aggregate byte events and concurrent HTTP file events feed one
  byte/rate/ETA progress bar. The callback uses a non-blocking `try_lock` so a
  busy terminal cannot serialize the transfer engine.

## Dependency and MSRV finding

`hf-hub 1.0.0` is pinned. Its published Xet dependency uses caret constraints,
which initially resolved the internally inconsistent Xet 1.6.0 family: that
source calls unstable `floor_char_boundary` on stable Rust 1.89. The manifest
and lockfile therefore pin the coherent release family `hf-xet`, `xet-client`,
`xet-core-structures`, `xet-data`, and `xet-runtime` 1.5.3, including lockless
`cargo install` resolution.

The native Xet stack depends on `redb 3` and `konst 0.4`, whose MSRV is Rust
1.89. hf2q's Cargo, README, and CI/release toolchain pins move together from
1.88 to 1.89. A lockfile that selects Xet 1.6 must not be accepted without a
fresh stable-toolchain compile and the live transfer proof.

## Evidence completed

- Rust 1.89 all-target/all-feature capped-lint check and optimized locked build
  passed. Eight pre-existing inference helpers that Rust 1.89 newly classified
  as production-dead remain explicitly documented as test/future-cache
  substrate; runtime behavior is unchanged.
- 74 focused Hugging Face downloader/resolution tests passed.
- 13 focused integrity tests passed.
- A live exact-origin metadata smoke test passed after resolving `main` to an
  immutable commit.
- A live native-Xet snapshot test used
  `hf-internal-testing/tiny-random-bert@f171d7.../pytorch_model.bin` (540,217
  bytes), proved the live `X-Xet-Hash` satisfies the production payload
  validator, downloaded only that selected file, preserved the exact snapshot
  parent, and passed hf2q's LFS SHA-256 verifier.

## Full-artifact performance proof

Do not derive a numeric speedup from the small correctness fixture. A live
v0.1.17 control was stopped after it had already reproduced the failure mode:
one established TLS connection, negligible CPU, and 4,186,326,178 bytes
retained after 1,699 seconds (2.350 MiB/s). The user explicitly declined
spending hours measuring a path already proven obsolete; the incomplete cache
is retained as diagnostic evidence and is not a release dependency. At that
observed rate the full artifact would take about 6,823 seconds (1h54m), but that
is an extrapolation, not a completed control result.

The checked-in release-grade harness is
`scripts/benchmark_hf_xet_download.sh`. It proves the selected production path
against the immutable 16,810,714,944-byte Qwen3.8 Q4_K_M artifact used by
hf2q's qualified onboarding path. Every run receives private Hub and Xet cache
roots, retains its evidence, verifies the final SHA-256, and reports the median
of three cold-cache native-Xet runs.

The v0.1.18 candidate binary
`72b7275acac023df3752d1ec0cc6b6cabebcd1935b840cf0eaabfc51c5d2f759`
completed the arm64/macOS 26.5.2 runs below. All three outputs were exactly
16,810,714,944 bytes with SHA-256
`1ee55c653644d6f645c6b2f39fc56a3ce28093620fd34dd43678875f348f2e1a`.

| Trial | Wall time | End-to-end throughput | Peak RSS |
|---:|---:|---:|---:|
| 1 | 551.57 s | 29.066 MiB/s | 3,153,494,016 bytes |
| 2 | 536.67 s | 29.873 MiB/s | 3,168,780,288 bytes |
| 3 | 642.31 s | 24.960 MiB/s | 3,324,559,360 bytes |
| **Median** | **551.57 s (9m12s)** | **29.066 MiB/s** | **3,168,780,288 bytes** |

The completed candidate median is 12.37x the retained-byte rate measured on
the stopped v0.1.17 control. This is a conservative end-to-end comparison: the
candidate timing includes native reconstruction and hf2q's required digest
pass, while the control rate is only partial-file growth and does not include
successful publication or final verification. Live inspection observed 64
simultaneous candidate CAS sockets at the adaptive ceiling, versus the old
client's single TLS connection.

The observable-transfer follow-up then ran the checked-in alternating-order
`scripts/benchmark_hf_xet_mode_ab.sh` against the same exact artifact. All six
outputs matched the expected bytes and digest. Adaptive trials were 250.12,
257.77, and 251.23 seconds; high-performance trials were 222.24, 225.99, and
260.04 seconds. Median throughput was 63.814 versus 70.941 MiB/s,
respectively. The retained adverse HP run prevents an overclaim: the preset
improved median time by 10.05% but did not win every trial. Its roughly 10.22
GB median RSS is the measured reason hf2q does not enable it below the
documented 64 GiB floor.

Reproduce it as:

```bash
scripts/benchmark_hf_xet_download.sh \
  /absolute/path/to/hf2q-candidate \
  /absolute/path/to/new-evidence-directory \
  3
```

The experiment contract is:

1. Use the exact repository, commit, filename, machine, network, and an empty
   isolated cache.
2. Require the payload to advertise Xet and capture logical bytes, wall time,
   median throughput, CPU, peak RSS, and failures.
3. Run at least three cold-cache trials of the release candidate.
4. Treat warm-cache lookup as a separate cache test, not network evidence.
5. Verify the identical final SHA-256 in every trial.

Acceptance should target link saturation and time-to-first-usable-model, not a
hard-coded multiplier. The anecdotal goal is to collapse a roughly one-hour
15 GiB transfer toward the few-minute class without weakening identity or
selection.

## Primary upstream references

- Hugging Face Hub environment and Xet controls:
  <https://huggingface.co/docs/huggingface_hub/main/package_reference/environment_variables>
- Hub migration to Xet architecture:
  <https://github.com/huggingface/blog/blob/main/migrating-the-hub-to-xet.md>
- Rust `hf-hub` client and native Xet support:
  <https://github.com/huggingface/hf-hub>
- Python `snapshot_download` worker model:
  <https://github.com/huggingface/huggingface_hub/blob/main/src/huggingface_hub/_snapshot_download.py>
- Xet client implementation:
  <https://github.com/huggingface/xet-core>
- Unsloth's checked-in Xet high-throughput/stall-retry downloader:
  <https://github.com/unslothai/unsloth/blob/main/.github/scripts/hf-download-with-retry.sh>
