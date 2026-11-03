# Model handoff feasibility — 2026-10-29

## Question and verdict

Can Qwen3.8 hand an ongoing coding session to DeepSeek-V4 and later resume it,
without repeating the historical prefill, within 30 seconds per switch
(10 seconds stretch), while retaining the complete chronological conversation?

**Verdict: a worthwhile research program, not an available translator integration.**
The core target is fast first-visit and repeated switching without repeating
the historical native prefill; reproducing any particular paper is not the goal.
Exact per-model checkpoint restoration plus replay of only newly added work is
the strongest initial engineering path for return visits. First-visit transfer
between these architectures requires new full-state transport research, fitted
artifacts, and task-level qualification. Neither path has an end-to-end runtime
latency or quality result in this investigation. The product target is unchanged;
suffix replay is explicitly not zero-prefill transfer.

## Evidence identities and boundaries

- hf2q source baseline: `7be222a5711dd9750b964e5555cdcfbece7dd2a6`.
- Clean source registry dependency: `mlx-native =0.15.14`, checksum
  `6126089f48c7a2170008f2da93e72ab139819f77630ef149d8ad4ebbd5abe1b4`.
  This describes the inspected source lock, not the unverified existing binary.
- SwaT source: `ae2de2285caf82f4d500526bf6e6992747b98eea`.
- LatentPort public evidence: `4adbbda3f749ba9f792d8f6020280a065d33f789`.
- OMP source: `31147f9d1a76608549821c3759f8487b4d66c72d`.
- Aither KVTransfer source: `bbee0d032e8462243d7f8ac58618b36131917b36`.
- Read-only model preflight used an existing binary reporting hf2q 0.1.27.
  Its source SHA was not independently established; its static output is not
  clean-source runtime qualification. Binary SHA-256:
  `3d91643a4eef90e0cd8d3b0ff99796c036c47fd30504387895b77b9f0865f71f`.
  No weights were loaded, hashed in full,
  converted, or downloaded. No inference server or release workflow was started;
  S8 subsequently started isolated protocol-fixture/client servers only.
- All code changes are research scripts/documentation in an isolated worktree.
  Raw receipts, upstream checkouts, and generated outputs remain local.

## Primary literature

### SwaT

[SwaT v1](https://arxiv.org/html/2610.15095v1) trains a directional transformer
mapper between frozen models, using cache reconstruction followed by receiver
generation supervision. Receiver tokens and character-span alignment remain
inputs. Its approximately 300M parameters and attention operations have nonzero
memory and execution cost. Summary comparisons additionally use Attention
Matching; that compression is a separate intervention. Experiments do not
establish repeated-transfer reliability or this project's production pair.

The [released source](https://github.com/Ar-Kareem/SwaT/tree/ae2de2285caf82f4d500526bf6e6992747b98eea)
provides training/evaluation code. `multiagent/model/unaware_comm.py` enumerates
all non-cross-attention layers as K/V layers, and `cache_layer` reads keys and
values. `cache_renderer.py` flattens per-token K/V and emits per-token receiver
K/V. Model aliases mentioning Qwen3.5 do not implement recurrent-state transfer.
The latency script explicitly excludes sender flattening, transfer, tokenization,
staging, model loading, compilation/capture, and decode, and assumes reusable
CUDA graphs. These exclusions matter to a complete Apple-Silicon switch.

At this pinned checkout no root license file or distributed trained translator
weights were found. Public code access alone does not establish redistribution
rights. Confirm permissions before adopting code or packaging derived weights;
independent implementation of the published method remains a candidate.

### LatentPort

[LatentPort v1](https://arxiv.org/html/2609.25053v1) studies one directed,
geometry-matched Qwen3.5 4B-Base to 9B-Base transfer. It includes recurrent and
convolution state; its selected combination directly copies those components
and translates attention K/V, then applies a learned correction. It reports
useful teacher-forced continuation but fails its near-native gate. Free coding
generation, cross-family transfer, longer-context qualification, and round trips
are unproven. Its timing omits translation and is not production switch timing.

The [public artifact repository](https://github.com/mmprotest/latentport-paper/tree/4adbbda3f749ba9f792d8f6020280a065d33f789)
contains an Apache-2.0 license and sealed observation files with verification
scripts. Its artifact coverage document identifies missing full rerun inputs,
including selected correction weights and paired state arrays. Reproducing
statistics is separate from reproducing inference or receiving a usable
translator. Locally, extraction followed by `analysis/verify_results.py`
completed successfully: **ALL CHECKS PASSED (202 checks)**, including the
sealed failed near-native gates. This verifies published observations and
statistics, not an independent GPU rerun. The initial system-Python invocation
failed for missing NumPy; rerunning in the existing developer environment
succeeded. No frozen evidence was edited.

### CacheLink

[CacheLink v1](https://arxiv.org/html/2610.15509v1) adds coupled K/V residual
transport with receiver-level supervision to an affine reconstruction baseline.
Its experiments show why better tensor reconstruction is insufficient evidence
of useful continuation. Tested pairs are attention-based, within model families;
they do not implement the Qwen-hybrid/DeepSeek-compressor interface. We found no
research-code link in the paper. Treat the method as a candidate to reproduce,
not an available dependency or a guarantee of native receiver quality.

### CacheBridge and available library screening

[CacheBridge v1](https://arxiv.org/html/2609.00891v1) is a useful lower-cost
baseline: restrict affine support by attention-head correspondence and fit
with attention-sensitive weighting. Its stated limitations explicitly leave
hybrid/sparse/sliding attention and open-ended repeated handoffs unestablished.
It is a calibration candidate, not evidence for the production pair.

The [Aither KVTransfer source](https://github.com/Aitherium/aitherkvcache/tree/bbee0d032e8462243d7f8ac58618b36131917b36)
was screened because a vendor article discusses DeepSeek transfer. Its actual
`geometry.py::check_pair` records latent-KV adapters as unimplemented, and the
self-test explicitly requires the MLA pair to be refused. Running
`python -m aither_kvcache.kvtransfer.geometry` locally passed with zero failures.
There is latent projection capture code, but that alone does not implement
hf2q's DeepSeek window/compressed/indexer/compressor state. No GDN state path
was found in the screened transfer modules. A partial capture/geometry mechanism
is not a usable Qwen3.8-to-DeepSeek translator. The library's wrong-document and
no-context controls are useful precedents for H4/H5, independent of its claims.

### Implications for our design

These are engineering hypotheses, not paper results:

1. Full state must be captured before testing learned portability. An incomplete
   same-model restore would invalidate any cross-model experiment.
2. Match the deployed quantized native runtime in the training/evaluation
   boundary. BF16 sibling results cannot certify a GGUF pair.
3. Optimize and score receiver behavior, including exact tool arguments and
   multi-turn task outcomes. Cache MSE is a diagnostic only.
4. Keep exact native checkpoints distinct from translated state. A later native
   suffix does not remove earlier approximation.
5. Reuse a returning model's independently retained exact prefix when possible.
   This avoids compounding translation errors, but does require suffix prefill.

## Source-bound hf2q inventory

| Surface | Existing mechanism | Missing for the product |
| --- | --- | --- |
| hf2q chat | `/model` activation and `ChatClient::set_model` | `set_model` currently resets the transcript; preserve one canonical conversation and provide explicit branching |
| Model lifecycle | `serve/api/lifecycle.rs::switch` drains leases, prepares spills, shuts workers, removes victims, loads replacement | Session-bound handoff transaction, persisted checkpoint acknowledgement and phase timing |
| Qwen | `inference/models/qwen35/kv_cache.rs`: full attention, GDN recurrent and convolution state, packed representations and MTP | Complete export/import contract and transport heads for all required persistent state |
| Qwen persistence | `kv_persist/families/qwen35_hybrid_persistor.rs`, codec 5; disk registry | Verify exact source artifact identity, publication boundaries, compact ownership and actual switch restore coverage |
| DeepSeek | `deepseek4/cache.rs`: window, compressed/indexer rows, main/indexer compressor states and cursors | A learned target/source representation covering all components and partial compression groups |
| DeepSeek persistence | `deepseek4/cache/anchor_image.rs` includes append-only rows missing from its in-memory recovery snapshot | Reuse complete image for eviction; never serialize only a recovery snapshot |
| Identity | Family fingerprints and rendered-token ledgers | Explicit full artifact, template, config, session, branch and provenance identity for cross-model handoff |
| Clients | Full-history OpenAI requests and native templates | Stable opt-in session identity, content reconciliation, capability discovery, model activation, telemetry |

DeepSeek's persisted namespace digest in
`kv_persist/families/deepseek4_anchor.rs::model_fingerprint_hex` hashes cache
geometry, compression schedule and quant label, rather than weight bytes.
It is not sufficient to bind a learned translator or a new exact handoff
checkpoint to one checkpoint's weights. This finding is a design input, not a
claim that existing callers have no additional isolation or a fix in this diff.

ADR-063 C1/#359 already owns exact multi-anchor validity and lineage work.
Reuse or depend on that work; do not attach this research to its active attempt
or duplicate its implementation. ADR-059 graft splice and artifact plumbing
are useful precedents, but a fixed graft is not a conversation translator.
ADR-042 governs DeepSeek semantics; ADR-013 governs Qwen's hybrid graph.

## Bounded spikes

### S1: execute the upstream layer selector

`scripts/research/model_handoff_spike.py` extracts and executes SwaT's unchanged
pure selector using Python AST, with a synthetic three-linear/one-full schedule.
It selects all four layers as K/V, including three that require recurrent state.
Source hashes are emitted. This falsifies an unchanged adapter, not the
possibility of training a full-state mapper.

### S2: persistent-state collision counterexample

The same script enumerates a small scalar delta-rule recurrence plus last-value
history. Prefixes `[0,1,0,0]` and `[2,0,0,0]` both yield recurrent state 0.125
and final value 0, while a target retaining the first value distinguishes them.
This illustrates why compressed state alone does not imply lossless historical
reconstruction. It is not a proof of a collision in the actual Qwen/DeepSeek pair.
Retained original text enables replay; it does not make learned caches exact.

### S3: exact-return protocol specification

The executable synthetic protocol reuses a 1,000-token native receiver prefix
and evaluates a 64-token appended suffix. Five identity changes, a rewritten
prefix, and translated-state-as-exact are rejected. This is a specification
probe; it implements neither runtime restoration nor model switching. Native
rendered tokens, not shared message counts, determine valid reuse.

### S4: real tiny hybrid cache negative control

`scripts/research/model_handoff_tiny_hybrid.py` constructs a four-layer,
random-weight Qwen3.5 model on CPU, with no downloaded weights. It compares a
complete copied cache to one with recurrent/convolution values zeroed, and
executes SwaT's unchanged reader against the actual hybrid cache object.
With PyTorch 2.11.0 / Transformers 5.5.0, seed 7, one CPU thread: complete
restore maximum absolute logit difference was **0.0**; zeroed recurrent/conv
state produced **0.013768933713436127** maximum difference. The unchanged
upstream reader raised `'LinearAttentionLayer' object has no attribute 'keys'`.
The reference uses its PyTorch fallback because optional accelerated linear
attention/convolution libraries are absent. This establishes mechanical
behavior of a small reference fixture only, not hf2q native parity, pretrained
quality, Metal execution, or cross-model success.

### S5: metadata-only capacity and latency sensitivity

Existing binary `info --ctx 32768 --max-slots 1` reported:

| Artifact | File | One-slot KV estimate | Model + KV estimate |
| --- | --- | --- | --- |
| Qwen3.8-27B vanilla Q4_K_M | 15.66 GiB | 1.67 GiB | 17.33 GiB |
| DeepSeek-V4-Flash-0731 hf2q-agentic-q2 | 100.05 GiB | 263 MiB | 100.31 GiB |

The host has 128 GiB total; static preflight reported approximately 91.5 GiB
available, below DeepSeek's estimate before scratch/allocator overhead.
No full-model load was attempted. The current main CI run was successful;
no release workflows were dispatched and no shared target build was run.

If activation must physically read all 100.05 GiB of DeepSeek, even a 30-second
budget requires 3.335 GiB/s with zero time left for anything else; ten seconds
requires 10.005 GiB/s. This is arithmetic, not measured disk throughput. mmap,
page cache and selective residency may change actual reads; measure page faults
and first useful output rather than assuming every activation reads the file.
A 300M BF16 mapper alone is about 0.559 GiB. Materializing a 64K-by-64K,
16-head BF16 score matrix would require 128 GiB, so a naive attention port
is inadmissible; fused/tiled attention and bounded temporary ownership are
necessary candidates to evaluate, not a demonstrated implementation.

The local conversion receipts identify the source revisions and declared GGUF
digests below. These are receipt provenance, not freshly verified full-file
digests in this investigation:

| Artifact | Source revision | Receipt-declared GGUF SHA-256 |
| --- | --- | --- |
| Qwen3.8-27B vanilla | `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0` | `cf0eb0103c1ffe7b3702670aa794a669bafbe328b95e892268305dd87c2c30d7` |
| DeepSeek-V4-Flash-0731 | `7872f01b1d1fe23eabc4c98b48bffcef5a386062` | `50df7e3dfe06321613c78075df4111b14985a0db31087992d7438f2895a3aad6` |

### S6: execute offline template-prefix stability controls

`scripts/research/model_handoff_render_probe.py` builds an isolated minimal
Rust package, imports the actual hf2q DeepSeek encoder source unchanged, and
runs its thinking mode on a synthetic prior assistant reasoning turn followed
by a new user turn. It also runs the pinned Qwen3.8 source Jinja through
Transformers' template helper. Cargo resolves locally/offline, then executes
with `--locked` in its own temporary target; no hf2q build target is shared.

| Fixture | Common UTF-8 bytes / old bytes | Whole old rendering survives? |
| --- | --- | --- |
| Qwen, preserve thinking | 1510 / 1510 | yes |
| Qwen, explicit removal | 1431 / 1510 | no |
| DeepSeek thinking, no tools | 119 / 209 | no |
| DeepSeek thinking, tool definitions present | 1358 / 1358 | yes |

Qwen's default `resolve_api_template_kwargs` inserts `preserve_thinking=true`;
therefore the review's general concern about dropped reasoning is a supported
variation, not proof that default Qwen always drops it. DeepSeek's encoder
disables `drop_thinking` when message tools are present. These fixture results
show why checkpoint placement and permitted transformations must be explicit.
They measure byte-prefix stability only, not rendered token LCP, generation
boundary stability, tool-result continuation or actual client wire behavior.
Native Qwen renderer parity is not established by using a Python Jinja helper.
S8 below adds installed-client protocol-fixture capture; native full-model
switch timing and GGUF-tokenizer qualification remain pending H0 work.
The probe pins observed behavior with hard assertions; it is a regression
control, not an unbiased survey. Its Qwen cases all include tool definitions.
The native template supports `preserve_thinking`; hf2q's default is part of
configuration identity and must match the full-replay control, rather than
being a non-native prompt modification made for this experiment.

### S7: token and synthetic generation-boundary survey

`scripts/research/model_handoff_prefix_probe.py` executes fifteen fixtures
using the same native Rust DeepSeek encoder and Qwen source Jinja as S6, then
tokenizes with explicitly supplied local HF `tokenizer.json` files. It
separates a completed history's re-rendering from a synthetic emitted token
sequence and an anchor before generation. No model generation occurs.

Selected measured cases, with tool definitions present:

| Configuration / fixture | Completed history → next request token LCP | Synthetic emitted history → next request token LCP |
| --- | --- | --- |
| Qwen preserve thinking, clean text | 373 / 373 | 373 / 373 |
| Qwen preserve thinking, surrounding whitespace | 373 / 373 | 343 / 376 |
| Qwen remove thinking, clean text | 341 / 373 | 341 / 373 |
| Qwen preserve thinking, tool result | 380 / 380 | 380 / 380 |
| DeepSeek thinking, surrounding whitespace | 332 / 332 | 332 / 332 |
| DeepSeek thinking, tool result | 356 / 356 | 356 / 356 |

Qwen's source template trims assistant content and reasoning. A stable
re-rendered completed history therefore does not prove that the originally
emitted token history survives: the whitespace fixture diverged immediately
after its 343-token generation prompt. That pre-generation anchor survived in
full when thinking was preserved. Without tools, DeepSeek's reasoning-removal
fixture retained 37 / 61 tokens; its 38-token generation prompt already changed
at the opening thinking marker. All fifteen synthetic token sequences decoded
to their declared text; joint retokenization matched in these fixtures.

These are HF-source tokenization controls, not hf2q's GGUF-built tokenizer,
actual sampled token IDs, native Qwen renderer parity, or checkpoint restore.
The Qwen tokenizer SHA-256 is
`0997f410c57a1f4e53b09e4be8f4a172d90edd9564368fb0847030937229b9f3`;
DeepSeek's is
`8f9f37ca37fdc4f5fd36d5cf4d3b0e8392edb4e894fd10cc0d70b4957c8633cf`,
from source revision `7872f01b1d1fe23eabc4c98b48bffcef5a386062`.
Transformers 5.5.0 / tokenizers 0.22.2 were used. Full local receipt:
`prefix-probe-h0.json`.

### S8: installed-client A-B-A protocol capture

`scripts/research/model_handoff_client_probe.py` runs installed clients with
temporary HOME/XDG/TMPDIR, a cleared environment, dummy credentials and a
loopback-only outbound sandbox. It captures actual Chat Completions HTTP
requests against a deterministic SSE fixture, not an inference backend.
OpenCode uses a private `--standalone` server and an explicit probe-owned
session; OMP uses one RPC process and awaited `set_model` operations.
No existing user session or background server is selected.

OMP 18.14.0, executable SHA-256
`608843959ab34295d52460dd09dd7d548b6c903c08a83c67954c2d4c36fd1f5a`,
completed Qwen → DeepSeek → Qwen: six HTTP requests, one tool round trip
per selected-model turn. Earlier visible assistant messages, user chronology,
paired call/result IDs and the tool-result marker survived. OMP's settled
state reported twelve messages. The custom host tool did not read user files.

Actual OMP requests change the system context's model identity and merge or
split system messages according to the selected model's capabilities. Foreign
reasoning moves from `reasoning_content` into visible `<think>` text, while
returning to Qwen restores its own earlier reasoning field. Tool IDs in this
simple fixture remain unchanged. No `prompt_cache_key` or custom session
header was sent. This proves why ledger reconciliation must be separate from
model-template alignment; the received request is authoritative.

Installed OpenCode is 2.1.1, not the initially inspected 1.2.27 reference;
executable SHA-256
`2f6fe5c3cf1e7827a93b14c7b07117ef0cde776c32613b0aae70c15e4780f271`.
The initial fixture incorrectly supplied the older `filePath` read argument;
the advertised installed schema requires `path`. A corrected request then
encountered `Permission denied: external_directory`. Neither failed setup is
client incompatibility or lost-context evidence. Raw failed receipts are
retained locally. The probe follows the advertised schema and restricts any
external-directory permission to its own temporary workspace.

The final corrected OpenCode 2.1.1 run also completed six HTTP requests with
one real marker-file read and tool-result continuation per selected-model
turn, retaining the earlier visible conversation and paired tool history.
The temporary workspace was canonicalized and the external-directory grant
covered only that workspace. Local passing receipt directories are
`omp-wire-h0` and `opencode-wire-h0-final`.

Protocol capture is not a live hf2q A-B-A inference run, quality gate, measured
switch latency, native token-prefix receipt, or proof of KV reuse. Compaction,
retry, branching, real coding tools and timeout behavior remain unqualified.

### S9: fresh no-weight Metal capacity observation

A direct default-device property query, without command queues, buffers or
weights, returned `recommendedMaxWorkingSetSize = 115448725504` bytes
(107.520004 GiB), `maxBufferLength = 86586540032` bytes and process-local
`currentAllocatedSize = 65536` bytes. The installed Metal SDK defines the
recommendation as an approximation for good performance, not a hard allocation
maximum. `iogpu.wired_limit_mb=0` and `wired_lwm_mb=0` are not unlimited-capacity
evidence. No OS policy was changed.

This does not admit DeepSeek: aggregate competing physical footprint, native
weight roots/padding, elementwise copies, boot-cache transfer branches,
scratch/pool lifetimes, mapper and retained-state peaks remain unclosed.
Removing all 263 MiB of the earlier 32K KV estimate would not eliminate its
8.84 GiB static shortfall. File size is a planning proxy, not measured runtime
residency. No full-model load, mapper Metal timing or latency run was attempted.

The latest published release observed was v0.1.27. Release run 40968373600
failed its candidate gate and did not publish a 0.1.28 candidate; other active
workflows/worktrees were left untouched. This investigation used no shared
hf2q build target. Local raw receipt: `h0-metal-capacity.json`.

## Client findings

At the pinned [OMP source](https://github.com/can1357/oh-my-pi/tree/31147f9d1a76608549821c3759f8487b4d66c72d),
`packages/ai/src/providers/transform-messages.ts` changes assistant/thinking
representation and tool IDs according to target-model/provider compatibility.
`openai-completions.ts` only emits `prompt_cache_key` under a capability policy.
An in-client session ID is therefore not proof of a stable server-visible
session identifier or byte-identical replay. Pin client versions and capture
actual requests; do not infer reuse from what a TUI displays.

The earlier audit used [OpenCode's model documentation](https://opencode.ai/docs/models/)
and did not execute clients. S8 now records installed-client protocol evidence
and its narrower limits. Native model switching, complete request-rewrite
qualification and actual hf2q inference remain outstanding.

## Reproduction

Use a developer Python environment with torch and transformers for the optional
tiny-model control; production hf2q MUST NOT invoke these scripts.

```sh
git clone https://github.com/Ar-Kareem/SwaT /tmp/swat-handoff
git -C /tmp/swat-handoff checkout ae2de2285caf82f4d500526bf6e6992747b98eea
python3 scripts/research/model_handoff_spike.py --swat-checkout /tmp/swat-handoff
python3 scripts/research/model_handoff_tiny_hybrid.py --swat-checkout /tmp/swat-handoff
# Optional offline source-render probe; requires Cargo's dependencies already
# cached and an exact Qwen3.8 source template supplied by the operator.
python3 scripts/research/model_handoff_render_probe.py --qwen-template /path/to/chat_template.jinja
python3 scripts/research/model_handoff_prefix_probe.py \
  --qwen-source /path/to/Qwen3.8-source \
  --deepseek-source /path/to/DeepSeek-source
# macOS installed-client capture; output must be a new directory outside the repo.
python3 scripts/research/model_handoff_client_probe.py \
  --client omp --binary /path/to/omp --output /tmp/h0-omp-wire
python3 scripts/research/model_handoff_client_probe.py \
  --client opencode --binary /path/to/opencode --output /tmp/h0-opencode-wire
```

The first script uses only Python's standard library. All print JSON receipts
and fail assertions on violated probe contracts. The tiny script uses random
weights and is not a substitute for native hardware gates. Follow the pinned
LatentPort repository's README to reproduce its statistical verifier separately.
No upstream paper text, datasets, weights, or generated receipts are added here.

## What remains to make this real

The complete staged architecture, exit criteria, client contract, artifact
distribution, ownership and release sequence are proposed in
[ADR-064](../adr/ADR-064-session-preserving-model-handoff.md). The expensive
unknowns are full-state transport quality, cross-template alignment, receiver
training capacity, total switch latency and repeated-switch degradation.
Public research does not supply a checkpoint-ready Qwen3.8/DeepSeek translator.
