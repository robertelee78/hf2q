# ADR-063 F1: proposed reachable peer-pin replacement

Draft for review, 2026-10-23. The initial investigation was read-only; later
sections record isolated builds and CPU conversion/parity probes. No product pin,
tracked fixture or shared checkout changed, and no inference runtime ran.

## Proposed decision

Replace the unrebuildable repository parity/quantization pin
`e15384a5cb092b080c2a01c0b9e3f8635079d6df` with reachable campaign source
`36e36e63a54fc8f85b95e9c499a022b597e478dd` **only after the compatibility
validation and review below**. Until then the latter is a proposed pin and an
identity-verified campaign executable, not a qualified repository parity oracle
or a demonstrated four-family runtime.

The mandatory latest-peer primary arm is separately resolved at each campaign's
start and frozen through that campaign (ADR-063 F1). Adopting this fixed pin does
not entitle future campaigns to call it latest. Neither pin authorizes replacing
hf2q conversion, quantization or production inference with an external runtime.

## Initial source and receipts (preserved observations)

- the product pin file named in `NOTICE`: old full SHA above. `NOTICE:5–10` identifies the
  upstream project, attribution/license, benchmark/parity role and pin location.
  There is no dedicated executable pin-update command or blanket automatic
  approval in these files; ADR-033 §§P0/P1 and its operator prerequisites govern
  fixture/quantization re-anchoring, ADR-063 F1 requires a reviewed proposal.
- `docs/adr/ADR-033-unified-quant-convert-pipeline.md:145–152,542,550` preserves
  earlier pin and fixture changes, including a stale-real-model-fixture finding.
  It requires re-anchoring gates and documenting rationale when peer source moves.
- `docs/adr/ADR-052-structured-output-compatibility.md:39–65` records the old
  object missing from non-shallow source, reflogs and unreachable scans and
  exact fetch returning `not our ref`. Its distinct grammar pin is
  `9cffdcc801582616250520966699cb5b25d28243`, not the product quantization pin.
- `scripts/fixtures/grammars/peer/PROVENANCE.md` records that separate grammar
  source and eight fixture hashes. Keep them; a product-pin change does not
  retag their historical source or license.
- `docs/research/apple-silicon-speed-parity-2026-10-22.md:16–28` records three
  conflicting surfaces: product pin e15384a5; shared checkout
  `970b51ce7c9ba5fff0726c5e9994f1c2f0e4eedc`; DeepSeek harness default
  `15586e2d7165570fb3aa7c26e0d442e289ef69de`. The absent default executable
  and stale build metadata b9278/5306f4b3b constitute no fourth valid binary.
- Today read-only checks confirm the shared peer checkout still at 970b51ce, clean;
  candidate object is not even in that checkout. The isolated checkout
  `/opt/hf2q-worktrees/adr063-peer-reference` is clean at 4231b23d.
- The archived `adr063-peer-build-receipt.json` records candidate source, binary SHA256
  `98ee8ebf20710ff9b1752b0a4807031b454e927e6aabbaf9e256205927178695`,
  configure/build commands, linked-library hashes and `model_loaded: false`.
  The archived `adr063-peer-version.txt` reports `0.7.0-dev (build 1, commit 4231b23)`
  with AppleClang21/Darwin arm64. Source HEAD and this version alone do not
  qualify parity; retain the receipt's binary/library identities.
- Upstream commit API successfully resolves commit `4231b23dc2c46661d01157aac5245bdcf03ce4b1`,
  dated 2026-10-23T12:56:59Z, subject `ggml : fix time overflow in ggml_time_us`.
  This establishes reachability today, not runtime/model support.

## Review and change sequence

1. Review this rationale under F1/#356 and record the explicit pin decision.
   Keep the product pin untouched during source/build/compatibility spikes.
   Preserve original receipts and any failed result; do not rewrite old ADR
   benchmark numbers or claim reproduction of an unavailable old executable.
2. Create an isolated hf2q topic worktree from current main and keep the shared
   the shared peer checkout checkout untouched. Cut all peer tools from the candidate
   source in isolated build directories. Record source/tree cleanliness,
   compiler/CMake/Metal/OS/hardware, flags, binary `--version`, SHA256 and actual
   dynamic dependencies for each tool, not just the already built server.
3. Inspect the source range where available, focusing GGUF metadata/tensor
   naming, quantization policies/codecs, tokenization/templates, grammar/tool
   semantics and target-family forward/cache changes. The missing old object
   prevents a trustworthy old→new code diff; document that limitation rather
   than invent ancestry or a synthetic old baseline. Use preserved fixtures as
   immutable behavioral evidence.
4. Validate the candidate as below. Only after clean results and independent
   review, change the product pin file named in `NOTICE` to the full candidate SHA. Update the
   current-pin statement in `tests/fixtures/ggml_quants/README.md`, append the
   dated ADR-033 pin-change rationale and validation evidence, and document the
   current distinction in the F1 identity manifest. Preserve May/September old
   hashes as historical provenance, including ADR-052 and grammar fixtures.
5. Do not silently edit the DeepSeek harness default as part of a one-line pin
   change: `scripts/benchmark_deepseek4_server_parity.sh:25` independently pins
   15586e2d. Record explicit campaign overrides for its source directory,
   executable and commit, or review a separate coherent default reconciliation.
   A new product pin does not make these defaults or absolute-path tests agree.
6. If ADR-033 or other managed planning sources change, inspect committed and
   working-tree bytes, regenerate immutable extraction, reconcile existing keys
   and verify Current source freshness. Commit/push/review/merge with focused
   evidence; keep pin and explanatory provenance together. Do not claim release
   or qualification from an unmerged proposal.

## Required validation before pin adoption

A. Build and hash candidate server, quantizer, needed raw-completion/bench tools
and reference harness libraries. Existing server-only build receipt explicitly
used examples/tests OFF: it does not qualify `llama-quantize`, conversion,
completion, synthetic fixture generation or the real-model conversion matrix.
No need to load a model for this initial tooling proof.

B. Re-run `scripts/ggml_quants_harness/build.sh` and `generate_all.sh` against
candidate-built libraries **in an isolated throwaway copy**, keeping seeds1/2,
NEON settings and all currently scripted types. The README says11 types but the
current script includes IQ4_XS too; inventory actual emitted cells, including
legitimate no-imatrix skips. Compare every generated byte/hash with tracked
fixtures. These scripts hard-code the shared peer checkout and output into the checkout;
review an explicit source/build/output parameterization or safely reproduce the
same compiler/link/generator commands with isolated paths. Do not follow the
old README's checkout command on the shared peer checkout. Capture linkage so
new headers cannot accidentally link old dylibs.

C. Run focused owned quantizer byte-comparison tests from a locked clean hf2q
build, then relevant conversion tests. For real-model confirmation use
`cargo test --release --locked --test byte_cmp_real_model -- --ignored byte_cmp --test-threads=1`
only after the harness paths and cache provenance are made explicit. Current
`tests/byte_cmp_real_model.rs:23–61` checks shared-checkout HEAD and hard-coded
quantizer path; mismatches return SKIP. A green process exit is not evidence
unless every required cell prints OK and none silently skipped. Its cached F16
reference filenames lack source identity (`ensure_canonical_f16`); use new
source-keyed isolated references rather than reusing old cached files. Likewise
`scripts/byte_cmp_full_pipeline.sh` reuses existing references and does not
establish their candidate-source identity.

D. Any changed fixture bytes are an investigation, not permission to overwrite
expected output until green. Identify codec/build/policy causes, compare owned
Rust behavior, and require a deliberate compatibility decision for each change.
Preserve pre-change fixture identities and document reviewed fixture replacement.
Do not relax parity/quality gates merely to accept a newer peer.

E. Qualify each intended model/tool runtime arm separately with identical
hf2q-produced GGUFs, pinned template/token sequences, exact sampling/context,
four-slot tool definitions, tool-result continuation, unary/SSE validity,
prefix reuse and semantic TTFT. Include matched settings and peer-best valid
settings with supported speculation; do not handicap the peer. Pin required MTP
and drafter artifact identities. Native-format comparisons stay separately
labeled. Run one full DeepSeek instance at a time after coordination/memory
preflight. Reachability and a successful `--version` are not this runtime gate.

F. Run the existing attribution ratchet `sh scripts/peer_ref_ratchet.sh` and
applicable locked repository checks. The CI job is `.github/workflows/ci.yml:42–52`.
The ratchet counts literal upstream-name lines outside its whitelist against
`data/peer_ref_baseline.txt`; it does **not** validate SHA reachability, binary
identity, compatibility, or runtime performance. Use “the peer” in new prose;
never raise its baseline to accommodate a pin change. NOTICE license policy
requires no change merely because a SHA changes.

G. If replacing frozen generated-text references, follow the substantive matched
prompt/settings/coherence checks in `docs/operator-peer-refresh-runbook.md`,
but do not execute its obsolete shared-checkout/stash or binary-discovery steps.
It targets historical b3d758750a with explicit raw-completion templates. Keep
old references/manifests and append candidate provenance; do not relabel them.
Any threshold change needs its own documented operator decision, not silent
fixture regeneration.

## Evidence level today

Reachable source and isolated server identity: observed. Historical pin
unrebuildable: already recorded source-bound research. New pin approved: no.
Quantizer/fixture evidence is updated below; the Gemma real-model conversion boundary is now validated below; wider adoption gates remain open. Four-family agentic runtime and
performance qualification: not run here. Proposal does not advance those states.


## Refreshed candidate and isolated codec validation

The initial 4231b23 executable above remains preserved. A later explicit upstream resolution selected `36e36e63a54fc8f85b95e9c499a022b597e478dd`; this is now the proposed replacement candidate. Its isolated server build/version identity is recorded in the F1 inventory. The shared checkout and product pin remain unchanged.

The candidate quantizer target also builds successfully. Its executable SHA-256 is `7e754481bdd1147506aaa28c87679c048b33788851d3cf3543d70a6bc3398ea2`. The help probe prints its usage and exits 1; that exit is recorded as help behavior, not mistaken for a successful quantization run.

For fixture validation, copied the existing `gen.c` and generation script into an isolated artifact directory, then compiled with the same C11/O2/NEON-enabled reference libraries and explicit candidate include/library/rpath locations. No tracked fixture or source script was rewritten, and no shared-checkout symlink/default was used. The generator's linked libraries point at the new isolated build. It generated all 48 input/expected files: 12 types, two importance variants, with input seed 1 and importance seed 2. There were no missing or skipped cells. Every generated file is byte-identical to its preserved tracked counterpart, including IQ4_XS omitted by the legacy README's 11-type count. This establishes the checked small-codec fixture boundary, not model-level equivalence.

The locked owned quantizer byte-comparison run verified all 24 fixture cells. Its broad test-name filter additionally returned a legacy real-model test as `ok` after that test printed a missing-fixture skip. Report it as **24 verified fixture cells and one skipped real-model cell**, not 25 independent parity passes. Full real-model conversion/cached-reference qualification under step C remains open; no skip may satisfy that obligation. No product pin is adopted and no four-family runtime claim follows.

The private comparison receipt freezes source, generator/library identities, every generated digest, exact byte comparisons, quantizer/help behavior and the owned-test log. Receipt SHA-256: `ffb28ed8503e77d4eed3f611a864716000683684e1819fcf11bf911a08f1d4a9`.


## Real-model validation update

The [full parity record](adr063-f1-parity-validation-2026-10-23.md) closes the deciding Gemma Q4_K_M conversion/tensor comparison: all 658 nonempty tensor entries have equal names, shapes, types and payload bytes under the existing canonical F16 staging contract. The frozen campaign Gemma artifact independently matches those 658 reference tensors too. Shared typed model-math/tokenizer metadata matches; descriptive/provenance metadata differs explicitly. This is not whole-file byte identity or a universal model-serving claim.

The declared reference dependency environment failed and is retained; a separately frozen Gemma-compatible Transformers 5.5.0 variant completed the probe. The unmatched F32-input experiment is a recorded failure, not an accepted codec regression or an excuse to loosen zero-byte acceptance. The checker now rejects structural/empty false positives, verified by six real GGUF controls. No legacy skipped test contributes proof.

Candidate source/build identity, 48 codec fixture files, 24 owned codec cells and this real-model boundary now support the proposed reachable replacement. Pin adoption still requires its separate review and any applicable wider conversion/runtime gates; the inventory task leaves the product pin unchanged.
