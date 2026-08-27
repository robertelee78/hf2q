# Hugging Face download state of the art — 2026-08-26

## Question

What is the fastest supportable way for hf2q to retrieve a hosted GGUF while
preserving hf2q's stronger integrity, cache, and publication contracts?

This note distinguishes source-backed capability from hf2q benchmark evidence.
The user report (about 59 minutes for 15 GiB through an older hf2q release and
under 3 minutes through an Unsloth workflow) is the incident trigger, not a
controlled benchmark.

## Conclusion

Use one retrieval implementation: `hf-hub` 1.x's native Xet path. Do not add
`curl`, a hand-written HTTP range scheduler, Git LFS, `hf_transfer`, or a Python
helper as an alternative downloader.

Native Xet is the current Hugging Face path for large files. It provides
adaptive concurrent range retrieval, parallel direct-address writes on
SSD/NVMe, resumable incomplete files, immutable content-addressed blobs, and
aggregate transfer progress. File-level batching is useful when several files
are needed, but it is not what makes one GGUF parallel: Xet reconstructs a
single large file through concurrent range work.

The native Xet result is necessary but not sufficient for hf2q. hf2q must also:

1. surface Xet's completed bytes, total bytes, rate, percentage, and ETA;
2. authenticate the completed immutable cache blob against its digest;
3. publish a final managed symlink to that blob atomically; and
4. avoid allocating or exposing a second payload-sized managed artifact.

## Existing hf2q baseline

The v0.1.18 release proof already completed three cold-cache downloads of the
exact 16,810,714,944-byte Qwen3.8 Q4_K_M artifact. Adaptive-default Xet measured
551.57 seconds (9m12s) median and 29.066 MiB/s end to end, including hf2q's
required digest pass. The stopped legacy v0.1.17 control retained bytes at
2.350 MiB/s through one TLS stream. Native Xet was therefore a proven 12.37x
improvement over the obsolete path, but the adaptive-default result did not
reach the user's under-three-minute comparison.

The v0.1.18 benchmark explicitly removed `HF_XET_HIGH_PERFORMANCE` and
`HF_XET_HP` from every run. That makes high-performance Xet the next smallest
measurable hypothesis. It does not justify enabling it before a resource and
throughput A/B.

Source: `docs/research/hf-download-rca-2026-08-26.md` and
`scripts/benchmark_hf_xet_download.sh`.

## Primary-source findings

### Xet supersedes the old alternatives

Hugging Face documents `hf_xet`/Xet as its optimal chunk-based transfer path.
All Hub repositories are Xet-enabled, `hf_transfer` support was removed in
`huggingface_hub` 1.0, and `HF_HUB_ENABLE_HF_TRANSFER` is ignored. The current
Rust `hf-hub` crate embeds the same Rust Xet client rather than invoking Python
or an external binary.

Sources:

- <https://huggingface.co/docs/hub/xet/using-xet-storage>
- <https://huggingface.co/docs/huggingface_hub/main/concepts/migration>
- <https://github.com/huggingface/hf-hub>

### One large file is already concurrent

`hf-hub` creates an Xet download group, queues the content-addressed file, and
lets xet-core reconstruct it. Xet uses adaptive download concurrency. The Hub
documentation describes parallel direct-address writes as the SSD/NVMe default;
sequential reconstruction writes are a special-case HDD tuning.

Therefore, replacing this with a single `curl` would regress the architecture.
Calling a multi-file API for a one-file GGUF would not create the missing
single-file parallelism; Xet already owns that layer.

Sources:

- <https://github.com/huggingface/hf-hub/blob/main/hf-hub/src/xet.rs>
- <https://huggingface.co/docs/huggingface_hub/main/package_reference/environment_variables>
- <https://huggingface.co/docs/hub/datasets-downloading>

### The progress data existed before the UI did

The Rust `hf-hub` Xet poller samples its download group every 100 ms and emits
`AggregateProgress { bytes_completed, total_bytes, bytes_per_sec }`, in addition
to per-file state. Its progress handler contract is thread-safe and intended to
keep callbacks cheap.

The v0.1.18 hf2q incident was not a lack of upstream telemetry. hf2q rendered
that telemetry on the helper child's stderr, redirected stderr to a private log,
and gave the parent only a one-shot "hosted download started" event. The parent
then rendered an elapsed-only spinner. The companion RCA traces that loss:
`hf-download-ux-rca-2026-08-26.md`.

Sources:

- <https://github.com/huggingface/hf-hub/blob/main/hf-hub/src/xet.rs>
- <https://docs.rs/hf-hub/latest/hf_hub/progress/index.html>

### High-performance mode is powerful but not a universal default

Xet defaults are adaptive and documented as sufficient to saturate most network
paths. `HF_XET_HIGH_PERFORMANCE=1` raises concurrency bounds and reconstruction
buffers. Hugging Face recommends it for high-bandwidth machines with at least
64 GiB RAM and warns that it can degrade performance on smaller machines.

The current documented preset changes download buffering roughly as follows:

| Setting | Default | High performance |
| --- | ---: | ---: |
| total reconstruction download buffer | 2 GiB | 16 GiB |
| per-file reconstruction buffer | 512 MiB | 2 GiB |
| reconstruction buffer hard limit | 8 GiB | 64 GiB |
| initial adaptive download concurrency | 4 | 16 |
| maximum adaptive download concurrency | implementation default | 124 |

Those memory costs rule out silently enabling the preset for every Apple
Silicon machine. The release decision must be based on an interleaved cold-cache
A/B on the target high-memory host. If the qualified-host policy wins, it must
be selected before the first Xet session is constructed and an explicit operator
setting must remain authoritative. This is one Xet implementation with a
resource policy, not a second downloader.

`hf-hub` currently constructs its cached native `XetSession` internally with
`XetSessionBuilder::new()` and exposes no client-builder hook for an injected
`XetConfig`. The supported integration point is therefore the upstream Xet
environment contract, applied once during hf2q startup before any background
thread or first session exists. Mutating process environment from a live
download callback is not an acceptable configuration mechanism.

Sources:

- <https://huggingface.co/docs/hub/xet/using-xet-storage>
- <https://docs.rs/xet-runtime/latest/xet_runtime/config/xet_config/struct.XetConfig.html>
- <https://github.com/huggingface/xet-core/issues/926>
- <https://github.com/huggingface/hf-hub/blob/main/hf-hub/src/client.rs>

The xet-core issue is useful measured evidence about memory cost, but it is an
open issue rather than accepted API documentation. The Hugging Face Xet guide
is authoritative for the >=64 GiB qualification and documented preset values.

### Chunk caching is workload-dependent

The Xet chunk cache defaults to disabled. Hugging Face states that this is often
faster for novel downloads; a >=10 GB chunk cache can help repeated related
revisions and deduplicated incremental workflows. hf2q must not turn it on
globally just because "cache" sounds faster. The immutable Hub blob/snapshot
cache remains active independently and is the payload hf2q publishes.

Sources:

- <https://huggingface.co/docs/huggingface_hub/main/package_reference/environment_variables>
- <https://github.com/huggingface/huggingface_hub/blob/main/docs/source/en/guides/manage-cache.md>

### What the Unsloth comparison establishes

Current Unsloth source delegates remote model retrieval to Transformers and
`huggingface_hub` APIs; it does not establish a separate proprietary transfer
protocol. Unsloth's download troubleshooting guidance recommends current
`huggingface_hub`/Xet and, for maximum-throughput environments,
`HF_XET_HIGH_PERFORMANCE=1` with the chunk cache disabled.

That supports the Xet choice. It does not make the anecdotal timing a valid
hf2q/Unsloth benchmark: cache warmth, selected artifact, region/CDN, process
environment, and high-performance settings were not controlled.

Sources:

- <https://github.com/unslothai/unsloth/blob/main/unsloth/models/vision.py>
- <https://raw.githubusercontent.com/unslothai/unsloth/main/unsloth/models/_utils.py>
- <https://unsloth.ai/docs/basics/troubleshooting-and-faqs/hugging-face-hub-xet-debugging>

### Xet still needs visible stall semantics

Unsloth's checked-in CI downloader records a real per-file Xet stall and wraps
the same `hf download` path in process-level termination and retry. This is
evidence that native Xet is not infallible. It is not a production algorithm to
copy literally: that script kills any process that has not exited after the
threshold, even if bytes are still advancing, so a legitimately slow link can
be reset forever.

hf2q's immediate contract is stronger observability: completed bytes and rate
must remain visible, and a stopped rate must be distinguishable from a slow but
advancing transfer. A future automatic stall retry must be based on *no byte
advance*, must have a bounded attempt/time budget, and must use a cancellation
boundary that can actually stop the in-flight Xet operation. A foreground Rust
thread cannot safely pretend a blocking Xet worker was cancelled while that
worker continues writing the cache.

Source:
<https://github.com/unslothai/unsloth/blob/main/.github/scripts/hf-download-with-retry.sh>

## Acceptance experiment — completed 2026-08-27

The experiment completed on an uncontended arm64 macOS 26.5.2 host with exact
candidate binary SHA-256
`069e3f3cb0fdb79ead4cfe3cc98e747cf33d02fd444b0be909904380ff352d07`:

```bash
scripts/benchmark_hf_xet_mode_ab.sh \
  /absolute/path/to/hf2q-candidate \
  /absolute/path/to/new-evidence-directory \
  3
```

Every output matched 16,810,714,944 bytes and SHA-256
`1ee55c653644d6f645c6b2f39fc56a3ce28093620fd34dd43678875f348f2e1a`.

| Round | Order | Mode | Wall time | Throughput | Peak RSS |
|---:|---:|---|---:|---:|---:|
| 1 | 1 | adaptive | 250.12 s | 64.097 MiB/s | 3,298,967,552 B |
| 1 | 2 | high performance | 222.24 s | 72.138 MiB/s | 10,157,277,184 B |
| 2 | 1 | high performance | 225.99 s | 70.941 MiB/s | 10,223,222,784 B |
| 2 | 2 | adaptive | 257.77 s | 62.195 MiB/s | 3,410,083,840 B |
| 3 | 1 | adaptive | 251.23 s | 63.814 MiB/s | 3,301,113,856 B |
| 3 | 2 | high performance | 260.04 s | 61.652 MiB/s | 10,897,260,544 B |
| **Median** | — | **adaptive** | **251.23 s** | **63.814 MiB/s** | **3,301,113,856 B** |
| **Median** | — | **high performance** | **225.99 s** | **70.941 MiB/s** | **10,223,222,784 B** |

The high-performance median wall time improved by 10.05% and used about 6.92
GB more median RSS. The slow 260.04-second HP trial is retained: the preset
improves the median but does not remove network variance. That result accepts
automatic HP mode only on Apple Silicon hosts with at least 64 GiB physical
memory. Adaptive remains the policy elsewhere, and either explicit upstream
mode variable wins.

1. Pin one exact public repository, revision, filename, size, and SHA-256.
2. Record hardware, OS, hf2q/hf-hub/xet versions, link-speed baseline, and free
   space.
3. Use a fresh `HF_HOME` for every cold run; do not delete shared caches.
4. Compare native Xet adaptive defaults with native Xet high-performance mode.
5. Interleave run order and report at least three completed runs per arm.
6. Report median wall time, payload throughput, peak RSS, CPU, and final digest.
7. Separately prove warm-cache activation and exact symlink publication.
8. Capture the actual hf2q PTY: bytes, total, percent, rate, ETA, completion, and
   bounded non-TTY output are release gates, not screenshots taken on faith.

CDN drift makes a favorable single run invalid evidence. A progress bar proves
observability, not speed; a fast transfer without final digest and cache-link
proof fails hf2q's correctness contract.

## Rejected paths

- **Single-stream `curl` or `reqwest`:** discards native Xet concurrency,
  content addressing, progress aggregation, and cache integration.
- **Legacy `hf_transfer`:** removed upstream and ignored by 1.x clients.
- **Python `huggingface_hub` subprocess:** same Xet engine with process and
  packaging complexity, contrary to the Rust-native boundary.
- **Git clone/Git LFS:** unnecessary repository metadata and an inferior API for
  selecting one hosted artifact.
- **Payload clone/copy/move into the managed store:** doubles disk allocation and
  disconnects hf2q from the authenticated Hub cache object.
- **Hard link publication:** filesystem-dependent and gives the managed store a
  second directory entry with lifecycle semantics that are harder to explain;
  the requested contract is an explicit symlink.
- **Blind global high-performance mode:** can reserve resources inappropriate
  for common 16–32 GiB Apple Silicon systems.

## Release decision record

The protocol decision is final: native Xet only. The exact-artifact A/B accepts
upstream high-performance mode on the >=64 GiB Apple-Silicon qualification and
adaptive defaults everywhere else. Progress, integrity, readable paths, and
symlink publication remain independent correctness gates; favorable throughput
does not compensate for a failure in any of them.
