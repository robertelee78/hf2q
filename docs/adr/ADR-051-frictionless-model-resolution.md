# ADR-051: Frictionless local-first model resolution

- **Status:** Accepted; implementation validated on the feature branch,
  release activation pending
- **Date:** 2026-08-23
- **Related:** ADR-005, ADR-018, ADR-033, ADR-045, ADR-046, ADR-047
- **Supersedes in part:** ADR-045's statement that onboarding does not create a
  no-options model workflow; ADR-045's installation and distribution decisions
  are unchanged

## Context

The public product journey should be complete after installation, setup, and
doctor:

```sh
hf2q chat jenerallee78/Qwen3.8-27B-Abliterated-SFT:Q4_K_M
```

The same model identity must work with `serve` and `convert`. Operators should
not have to remember whether bytes came from a prior native conversion, a
hosted GGUF download, the hf-hub cache, or a manual download. They should not
repeat a multi-gigabyte transfer merely because the artifact is outside one
legacy cache manifest.

The baseline spike on 2026-08-23 used the existing debug binary from source
commit `b2d2ae0f`. All three intended commands failed in Clap before resolver
code ran:

| Command shape | Exit | Observed failure |
|---|---:|---|
| `hf2q serve owner/repo:Q8_0` | 2 | unexpected positional argument |
| `hf2q chat owner/repo:Q4_K_M` | 2 | unexpected positional argument |
| `hf2q convert owner/repo:Q8_0` | 2 | required `--output` missing |

This falsifies the narrow hypothesis that the problem is only a missed cache
lookup. The shared operand grammar, artifact authority, managed layout, and
fallback rules are one product contract.

A metadata-only spike against the operator's concrete repository resolved
commit `0a72776892f98db49381fdf69f4b9982222ec9dc`. It exposes selectable Q4,
Q5, Q6, and Q8 text artifacts plus two projector companions: one exact
Q4-text-stem companion and one generic Qwen projector. The operator's managed
root already contains Q5, Q6, and Q8 files whose byte lengths exactly match
the immutable Hub inventory. This measured case reformulated two details:

- Loose discovery scans once, orders candidates by modification time, and
  hashes newest-first until the first exact Hub digest match. It does not hash
  every multi-gigabyte candidate before making the recency choice.
- Projector matching prefers an exact `<text-stem>-mmproj.gguf`, then one
  generic `mmproj-*` companion. Multiple candidates after those rules remain
  ambiguous and produce the documented text-only warning.

ADR-047 already supplies useful mechanisms: receipt-backed local discovery,
exact hosted metadata, immutable revisions, strong LFS digests, bounded
verification, and a safe hosted activation path. It does not make canonical
model identities a first-class CLI operand, does not scan the canonical data
directory by default, and deliberately kept ordinary serve conversion-first.
This ADR changes that product policy without weakening ADR-047's authority
checks.

## Decision

### 1. One model operand grammar

`serve`, `chat`, and remote `convert` accept:

```text
owner/repository
owner/repository:QUANT
```

`QUANT` is case-insensitive at input and normalized to its canonical GGUF name.
An existing path, an absolute path, or an explicit `./`, `../`, or `~` path is
always a path. The literal `list` is reserved for the local inventory in
`serve` and `chat`. `--model` remains a compatibility spelling; supplying both
spellings is an input error.

For `convert`, `owner/repository:QUANT` and `--quant QUANT` are equivalent.
Different values fail before network or filesystem mutation. Hosted
pre-quantized bytes never satisfy `convert`: conversion still downloads source
weights and runs hf2q's native converter and quantizer.

### 2. Managed layout and identity receipt

The default durable root is:

```text
${XDG_DATA_HOME:-$HOME/.local/share}/hf2q/models
```

Repository artifacts live under:

```text
<root>/<owner>__<repository>/<immutable-revision>/<artifact>
```

Each adopted, downloaded, or converted artifact has bounded sidecar authority
that records the repository, immutable revision, exact artifact bytes, SHA-256,
quant identity, origin, materialization time, and last successful use. A
projector binding additionally records its own immutable filename, bytes, and
SHA-256. Schema-v3 conversion receipts and the existing canonical `ModelCache`
remain valid authorities and are merged into the same inventory.
Inventory admits a sidecar row only while its text artifact remains a
non-symlink regular file of the recorded size; projector availability uses the
same cheap physical check. Listing does not hash payloads.

`--output FILE` keeps its exact-file meaning. `--output DIR` places the derived
artifact name in that directory. Without `--output`, `convert` writes the
revision-bound managed path. Serve/chat hosted downloads use the managed path
unless their command supplies an explicit output destination.

### 3. Local-first resolution without filename trust

Resolution is deterministic:

1. Find verified, repository-bound local candidates in schema-v3 conversion
   receipts, managed-artifact sidecars, and the canonical `ModelCache`.
2. If an exact quant was requested, use only that quant. Otherwise prefer the
   compatible candidate with the newest successful-use timestamp; if none has
   use history, prefer the newest verified materialization timestamp.
3. Scan bounded configured roots for loose GGUFs. A loose artifact is adopted
   only after exact Hub metadata supplies an immutable revision, filename,
   byte length, and strong SHA-256 and a complete local digest matches. A
   filename, substring, modification time, or quant-looking suffix is never
   authority.
4. If no local candidate wins, query hosted metadata. An exact quant downloads
   exactly that supported quant after disk preflight. An unqualified request
   chooses the setup/live hardware recommendation, then the nearest lower
   supported hosted tier. Unsupported or ambiguous hosted choices fail with
   the available exact options.
5. If no supported hosted GGUF exists, `serve` and owned local `chat` fall back
   to hf2q native source conversion. `convert` always uses this source path.

Adoption never moves or deletes operator bytes. hf2q hard-links into the
managed layout when possible, copies atomically across filesystems, then writes
authority only after destination size and SHA-256 pass. A matching existing
destination is reused. A conflicting destination fails closed and is never
overwritten implicitly.

Hosted disk preflight models the actual materialization plan. Hub cache and
managed destination on one filesystem require one model-sized allocation and
use a hard link; cross-filesystem placement preflights the second allocation
before payload transfer. An already cached exact-size artifact does not fail a
new-download space check merely because the cache is now relatively full.

Compatibility means the current runtime supports the GGUF architecture and
file type, the artifact passes header and integrity checks, and existing disk,
memory/admission, and pair preflights accept it. Recency never overrides those
checks.

### 4. Quant defaults and idempotent conversion

Remote conversion quant precedence is:

1. a non-conflicting operand suffix or explicit `--quant`;
2. the setup-recorded recommendation; or
3. live hardware selection when setup has not run.

A matching verified hf2q conversion at the final destination is an idempotent
success and does not re-download or re-quantize. Default multimodal conversion
is a no-op only when both text and projector receipts, digests, and the pair
generation/binding contract verify; a prior text-only output cannot silently
satisfy the paired default. An external or hosted GGUF is never reclassified
as a completed conversion.

### 5. Multimodal companion behavior

After selecting text GGUF bytes, hf2q determines whether the model declares a
supported multimodal architecture. A trustworthy locally bound matching
projector is loaded automatically. If it is absent and the exact repository
revision exposes one unambiguous matching `mmproj` companion, hf2q downloads,
verifies, binds, and loads it before inference.

Automatic companion failure is a visible warning followed by text-only
serving. It never substitutes an unrelated projector. An explicitly supplied
`--mmproj` retains fail-closed behavior. The existing text/projector pair
preflight remains the final authority; a failed automatic pair preflight warns
and falls back to text-only.

The same local behavior applies to an explicit GGUF path. hf2q first uses a
valid managed sidecar or matched conversion receipt, then the text GGUF's
bound projector digest, then an unambiguous sibling projector. It never sends
a local-path model to the Hub. Ambiguity or failed pair admission warns and
continues text-only.

### 6. Chat and inventory behavior

`hf2q serve list` and `hf2q chat list` call the same read-only inventory and
show repository, revision, quant, origin, recency, projector availability, and
path. Listing performs no Hub request and no full-file hashing.

`hf2q chat owner/repository[:QUANT]` uses the same preparation resolver as
serve by starting an owned loopback server with that exact operand. Existing
DNS-SD advertisements do not carry immutable repository/revision/quant
identity, so targeted chat deliberately does not risk reusing an unrelated
large resident model. Plain `hf2q chat` still discovers and reuses existing
servers. The owned child inherits `HF2Q_AUTH_TOKEN`; chat sends that token only
after the discovery PID matches its own child, and prints a bounded heartbeat
while first-use download or conversion continues. The TUI does not create a
second downloader or converter.

### 7. Concurrency and failure safety

Per-repository/revision/quant locks cover adoption, download, conversion, and
sidecar publication. Successful-use publication reloads the cache manifest
under that lock rather than flushing a stale process snapshot. Shared
projector publication rechecks an exact destination after a concurrent
`EEXIST` winner. Every expensive path rechecks after acquiring its lock.
Downloads and copies use same-directory temporary files and atomic rename.
Disk space is checked before transfer using the exact hosted byte count or the
native conversion plan. Interrupted work leaves no authoritative partial
artifact. Zero-byte, symlinked, non-regular, digest-mismatched, stale-revision,
or unsupported candidates never win resolution.

## Acceptance gates

- CLI tests prove all approved operands, compatibility spellings, conflict
  failures, and no-output conversion parsing.
- Pure resolver tests prove exact-quant selection, use/materialization recency,
  local-over-hosted precedence, ambiguity, and nearest-lower hosted fallback.
- Filesystem tests prove bounded loose discovery, exact digest adoption,
  same-filesystem hard-link, cross-filesystem copy fallback, atomic sidecars,
  conflicting-destination refusal, and interrupted-partial exclusion.
- Download tests prove immutable metadata revalidation, exact quant, disk
  preflight, full SHA-256, and unique companion selection without payload on
  ambiguous or declined paths.
- Conversion tests prove setup/live quant precedence, suffix/flag conflict,
  managed revision output, receipt-backed no-op, and that hosted GGUF never
  satisfies convert.
- Serve/chat tests prove one shared inventory, owned-chat reuse of serve
  preparation, automatic projector load, text-only warning fallback, and
  explicit-projector fail-closed behavior.
- Locked focused tests, all-target check, release build, full hosted-safe test
  suite, and nonzero Rust coverage evidence at the exact branch head.
  Agentic-QE complexity/audit output is advisory because the installed tool
  does not instrument Rust coverage; a zero-test or heuristic-only report is
  never represented as a passing gate.
- A real operator-owned local artifact is detected without payload transfer;
  any real model load is run only on an uncontended Apple Silicon host and is
  reported separately from source/unit proof.

## Consequences

The common path becomes short and deterministic, and already-owned bytes win
without weakening integrity. The managed sidecar is additional indexed state,
not a trust shortcut: source-bound conversion receipts, exact Hub metadata,
and complete digests remain the authority.

Metadata lookup may be required to adopt a manually downloaded loose artifact.
That small request is intentional because avoiding payload duplication without
inventing identity from filenames requires remote immutable evidence. If the
Hub is unavailable, already-bound artifacts remain usable while unbound loose
files stay visible but ineligible.

ADR-047's explicit diagnostic activation remains supported for remote or
pre-existing endpoints. This ADR adds the simpler owned-local path; it does not
weaken multi-model admission, process ownership, or OpenAI compatibility.

## Validation evidence

The pre-refinement Apple Silicon spike used the immutable Qwen3.8 repository
revision `0a72776892f98db49381fdf69f4b9982222ec9dc`. A bare repository operand
selected the newest verified local Q6 artifact, adopted it into the canonical
managed directory without moving the operator file, retrieved and loaded the
matching projector, and reached readiness. The managed text and projector
were hard links to the already-owned exact inodes. `serve list` and `chat
list` reported the bound Q6 plus the other locally owned unbound quants without
Hub traffic.

That loaded artifact passed the canonical Qwen multi-turn agentic tool/cache
script and the first-image-after-text vision script. The post-review branch
adds blocking focused tests for operand parsing, exact/nearest-lower quant
selection including Q2, local recency, loose-digest disambiguation,
destination reuse/conflict, receipt-backed pair reuse, direct-path projector
resolution, concurrent publication, hard-link mutation detection, targeted
chat authentication/heartbeat, and stale inventory exclusion. Exact final
commands and release SHA are recorded by the release workflow rather than
claimed by this ADR before publication. Focused Rust coverage from
`cargo llvm-cov --locked --bin hf2q --all-features --json --summary-only --
serve::managed_artifacts::tests::` ran all 18 resolver tests; the directly
exercised managed-artifact modules reported nonzero line coverage ranging from
22.81% for inventory formatting to 92.71% for receipt/storage validation.
