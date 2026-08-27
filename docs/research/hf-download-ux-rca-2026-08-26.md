# Hugging Face download UX RCA — 2026-08-26

## Executive summary

hf2q v0.1.18 fixed the payload transport bottleneck by moving large model
downloads to `hf-hub` 1.0's native Xet path, but the release did not deliver a
usable download experience for owned `hf2q chat` startup.

For the exact command and artifact below, the interactive parent displayed a
spinner and elapsed time for more than five minutes without transferred bytes,
percentage, rate, or ETA:

```text
hf2q chat jenerallee78/Qwen3.8-27B-Abliterated-SFT:Q4_K_M
qwen38-abliterated-sft-hf2q-q4_k_m.gguf
16,810,714,944 bytes (15.7 GiB)
revision 0a72776892f98db49381fdf69f4b9982222ec9dc
```

The transfer was making progress. `hf-hub` emitted native Xet aggregate byte
events approximately every 100 ms, and hf2q consumed those events. The visible
chat process could not receive them. The byte progress adapter was attached to
an indicatif bar owned by the server child, while owned chat intentionally
redirected that child's stderr to a private log. The only structured event sent
to the parent was the one-time `HostedDownload { filename, bytes }` start
event. The parent therefore selected its indeterminate spinner template, whose
only changing numeric field is `{elapsed_precise}`.

This is a product and release-gate failure, not an upstream Xet limitation.

The same journey exposed two adjacent storage UX failures:

- the managed revision directory was created before the artifact was
  published and remained visibly empty during the cache download;
- the directory name was an internal `v2-<hex(owner/repository)>` storage key,
  and the post-download publication path cloned the cache blob or copied it on
  clone-unsupported/cross-filesystem destinations instead of publishing a
  managed symlink to the already-authenticated Hugging Face cache blob.

The unrelated `SIGTERM` observed at the end of the reproduction was issued by
another concurrent agent and is not attributed to hf2q's download code. It is
tracked as a host-coordination incident, not part of this code-cause chain.

## User-visible contract

Every model-sized payload transfer must expose one authoritative progress
surface owned by the foreground process. Within one second of the first payload
byte, an interactive terminal must show:

```text
Downloading qwen38-...-q4_k_m.gguf  4.9 GiB / 15.7 GiB  31%  29.4 MiB/s  ETA 6m 04s
```

The bar must advance monotonically and include:

- logical bytes completed and total bytes;
- percentage;
- smoothed logical throughput;
- ETA after rate warm-up;
- the current phase: metadata, download, integrity verification, link
  publication, or model load;
- a useful stalled state when no byte update has arrived for a bounded period.

Non-TTY output must remain line-oriented and bounded. It should emit throttled
milestones rather than terminal control sequences. A cache hit must explicitly
say that no payload transfer was required.

For the default managed store, a successful hosted download must leave a
human-readable logical path such as:

```text
~/.local/share/hf2q/models/
  jenerallee78/
    Qwen3.8-27B-Abliterated-SFT/
      0a72776892f98db49381fdf69f4b9982222ec9dc/
        qwen38-abliterated-sft-hf2q-q4_k_m.gguf ->
          ~/.cache/huggingface/hub/models--jenerallee78--Qwen3.8-27B-Abliterated-SFT/blobs/<digest>
```

The exact repository, revision, size, SHA-256, quant, cache target, and origin
remain sidecar authority. A readable path is presentation, not identity.

## Actual control flow

```text
hf-hub/Xet worker tasks (~100 ms aggregate events)
        |
        v
src/progress.rs::HubDownloadProgress
        |
        v
server-child indicatif byte bar
        |
        v
child stderr -> private hf2q-chat-server-*.log   (not visible)

resolver -- one HostedDownload start event --> Unix datagram --> chat parent
                                                        |
                                                        v
                                             indeterminate spinner
                                             + elapsed time only
```

There were two progress systems for one operation. The system with byte data
had no route to the user, and the system with terminal ownership had no byte
data.

## Direct technical causes

### 1. Progress was rendered at the wrong ownership layer

`download_hub_artifact` constructed a new, private `ProgressReporter` and
passed its `HubDownloadProgress` handler to `download_file`. That adapter
correctly handled `DownloadEvent::Start`, Xet `AggregateProgress`, per-file
`Progress`, and `Complete`, but its only output was an indicatif bar.

Owned chat starts a server child with stdout disabled and stderr redirected to
a private temporary log. This is intentional: server tracing must not corrupt
the foreground chat UI. It also means a child-owned progress bar is inherently
unobservable.

### 2. The structured startup protocol had no byte-progress event

`StartupEvent` contained `HostedDownload { filename, bytes }`, a start event
only. It had no download equivalent to the existing `VerifyProgress` event.
The server could therefore send the selected artifact and total size but not a
changing byte position.

### 3. The parent explicitly reset to an indeterminate style

On `HostedDownload`, `StartupUi` printed one history row and called
`reset_spinner`. The spinner template contains `{elapsed_precise}` and no
position, length, percent, throughput, or ETA fields. The screenshot is the
literal intended output of that branch, not an indicatif rendering defect.

### 4. Transfer and presentation were coupled

The progress adapter owned a concrete terminal bar rather than publishing
output-agnostic progress snapshots. That made it easy to verify the adapter in
isolation and impossible to reuse the data across the child/parent boundary.

## Storage and inspectability causes

### Empty managed revision directory

The payload is downloaded and atomically published by `hf-hub` under its
standard cache. hf2q does not place partial GGUF bytes in the managed store.
For multimodal planning, however, destination-parent authority is retained
before the text/projector transfer. This creates the managed revision
directory before either final artifact exists. The result is safe but
misleading: the only user-visible destination is empty while the real bytes
are accumulating elsewhere.

The fix is not to expose a partial GGUF. The fix is to report the actual cache
phase/path and defer the visible revision directory until final link
publication.

### Opaque repository directory

`managed_revision_dir` writes `v2-` followed by the lowercase hex encoding of
`owner/repository`. The encoding is injective and case-fold stable, but it is
an internal storage key presented as the primary human filesystem interface.
Sidecar identity already carries the exact repository and revision, so the
hex key is not required for user comprehension. Existing v2 paths remain
read-compatible; new writes should use bounded, validated owner/repository
components and use an internal digest only for an actual collision.

### Duplicate materialization policy

After authenticating the HF cache artifact, `materialize_hosted` calls
`materialize_preverified_exact`. On APFS this attempts a copy-on-write clone;
if cloning is unavailable it copies and hashes the full payload. This avoids
duplicate physical blocks on the common same-volume case but still creates an
independent managed regular file, can consume a second full allocation, and
adds a large post-download phase on fallback filesystems.

For hf2q-owned hosted downloads, the standard HF cache blob is already the
single exact payload authority. The default managed entry should be an
atomically published final-leaf symlink to the authenticated canonical blob,
with a regular no-follow sidecar. hf2q must retain and revalidate the target
descriptor before activation. If the operator clears the HF cache, a dangling
managed link is a cache miss only when its stored link text is the exact
digest-named repository blob hf2q would publish. After redownload and digest
verification, that same link becomes valid again. hf2q must never silently
accept or repair an arbitrary target.

## Why v0.1.18 tests passed

| Gate | What it proved | What it could not prove |
|---|---|---|
| `HubDownloadProgress` unit test | mixed Xet/per-file positions are monotonic and not double-counted | used `ProgressBar::hidden()`; rendered no terminal output and never crossed the chat protocol |
| live tiny Xet test | exact selection, Xet identity, snapshot path, and digest | opt-in, tiny, non-PTY, and did not exercise owned chat |
| 16.81 GB cold-cache benchmark | exact artifact integrity and median transport throughput | redirected stdout and stderr to files; invoked the hidden fetch helper, not `hf2q chat` |
| startup telemetry tests | typed bounded datagrams and verification progress | schema had no download-progress event, so absence was treated as complete behavior |
| frictionless packed-binary PTY smoke | banner, terminal protocols, lifecycle, local reuse | used no-network/local fixtures and never performed a long hosted download |
| protected release workflow | exact artifact, package, notarization, and the declared gates | inherited the same missing operator-facing acceptance contract |

The benchmark's redirection was appropriate for reproducible timing, but it
was incorrectly treated as sufficient release proof for an interactive
feature. Performance and UX needed separate gates.

## Five-whys analysis

1. **Why did the user see only elapsed time?** The foreground parent rendered
   an indeterminate spinner.
2. **Why was it indeterminate despite known total bytes?** The parent received
   only a one-time start event.
3. **Why did it not receive Xet byte updates?** Those updates mutated a bar
   owned by the hidden server child.
4. **Why was the progress adapter child-renderer-specific?** The Xet migration
   added a terminal sink directly to the downloader instead of defining a
   transport-neutral progress state and routing it through the existing
   startup channel.
5. **Why did release validation not catch that?** The acceptance plan tested
   transfer speed, integrity, hidden-bar state, and general PTY lifecycle as
   separate units, but never tested the literal cold-cache `hf2q chat` journey
   with advancing bytes visible in the packed binary.

The root organizational error was validating subsystem facts instead of the
complete user journey.

## Corrective design

### One progress state, one foreground renderer

The hf-hub handler should update a thread-safe, output-agnostic snapshot:

```text
phase, filename, completed_bytes, total_bytes, bytes_per_second,
started_at, last_progress_at, complete
```

The synchronous resolver can run the blocking download on a scoped worker and
poll or receive coalesced snapshots at a bounded rate. Its existing startup
callback emits typed `DownloadProgress` events. Owned chat sends those events
over the private datagram; direct serve feeds the same event to its local
`StartupUi`. Only the foreground renderer owns indicatif.

The Xet callback must remain nonblocking. Coalescing is required because
upstream emits at roughly 10 Hz and startup telemetry is explicitly
best-effort. Dropping an intermediate snapshot is safe; dropping the newest
state forever is not. Completion and failure remain authoritative through the
download result, not telemetry.

### Symlink publication contract

For a default hosted destination:

1. resolve one exact revision and artifact;
2. download into the standard HF cache through Xet;
3. verify exact size and SHA-256;
4. canonicalize and retain the repository blob through no-follow directory
   authorities;
5. create a temporary final-leaf symlink inside a hidden publication location;
6. atomically publish the link without replacing conflicting authority;
7. write the regular sidecar only after target/link identity checks pass;
8. activate through a retained descriptor, not by trusting a later pathname
   lookup.

No model-sized copy, clone, hard link, or move belongs in this path. Explicit
native conversion outputs remain independent hf2q-produced files; this change
does not turn conversion into a pre-quantized download.

### Human-readable layout and compatibility

New default writes use `<owner>/<repository>/<revision>/<artifact>`. Repository
input is already bounded ASCII with one owner/repository separator and safe
component characters. Existing v2 and legacy layouts remain read-compatible
through their bindings. Lock identity remains based on the exact canonical
repository string, not the display path. On a proven case-fold collision, hf2q
may append a short digest to the readable repository component and must say so
in inventory output.

## Prevention gates

The next release cannot claim this fixed without all of the following:

1. A pure event test drives `Start`, Xet aggregate updates, per-file updates,
   and `Complete`; snapshots stay monotonic and expose total/rate state.
2. Startup wire tests accept valid download progress and reject zero totals,
   completed bytes above total, unsafe filenames, and oversized frames.
3. Parent UI tests prove a hosted start selects a byte bar, updates position,
   includes percentage/rate/ETA, and transitions to verification/publication.
4. A child/parent datagram test proves byte updates survive the exact owned-chat
   boundary while child stderr remains private.
5. A non-TTY test proves throttled line output contains byte totals, percent,
   rate, and ETA without ANSI controls or unbounded event spam.
6. Filesystem tests prove a downloaded cache snapshot publishes a final-leaf
   symlink to the authenticated blob, allocates no second payload, refuses a
   conflicting link, detects retargeting, and recognizes an expected dangling
   link as repairable only through exact re-download and digest proof.
7. Layout tests prove readable new writes and read compatibility for existing
   v2 bindings without renaming or deleting user data.
8. The packed release binary runs the literal `hf2q chat` cold-cache journey
   through a PTY or deterministic Xet event fixture and captures at least two
   strictly advancing visible byte positions before completion.
9. One uncontended real 16.81 GB cold-cache Xet run records the same artifact,
   revision, cache roots, terminal width, multiple progress observations,
   final digest, elapsed time, and median transport benchmark. A redirected
   throughput benchmark alone is insufficient.

## Local acceptance evidence (2026-08-27)

The corrected tree completed the literal owned-chat journey with fresh data,
Hub, and Xet roots using
`jenerallee78/Qwen3.8-27B-Abliterated-SFT:Q4_K_M`. The selected artifact was
16,810,714,944 bytes at revision
`0a72776892f98db49381fdf69f4b9982222ec9dc`; it completed in 4 minutes 24
seconds, passed SHA-256
`1ee55c653644d6f645c6b2f39fc56a3ce28093620fd34dd43678875f348f2e1a`,
loaded, reached the chat prompt, and exited cleanly via `/quit`.

The resulting managed leaf used the readable owner/repository/revision layout,
was a symlink to the canonical digest-named Hub blob, and resolved to the same
inode (`131573324`) as that blob. The projector leaf was also a Hub-blob
symlink. A recording-terminal test drove the real interactive renderer and
asserted visible cache-to-managed-link context, completed and total bytes,
percentage, rate, and ETA. Focused tests cover aggregate progress, typed wire
validation, bounded non-TTY output, readable paths, legacy-v2 reads, exact
cache-link reuse, dangling-link repair, retarget refusal, and one-extent disk
planning. The protected workflow remains the authority for the final v0.1.19
packed and signed bytes.

## Scope boundary

The v0.1.18 Xet transport, exact revision pinning, allow-listed snapshot,
strong digest checks, atomic HF cache publication, and conversion/inference
ownership boundaries remain correct. This RCA changes progress routing,
managed hosted-artifact publication, filesystem presentation, and their
release gates. It does not add alternate payload transports or weaken
integrity to obtain a prettier UI.
