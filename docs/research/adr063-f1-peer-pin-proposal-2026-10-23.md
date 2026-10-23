# ADR-063 F1: proposed reachable peer-pin replacement

Draft for review, 2026-10-23. No product pin, fixture, checkout, code, or GitHub
state changed by this investigation. No build or model load performed.

## Proposed decision

Replace the unrebuildable repository parity/quantization pin
`e15384a5cb092b080c2a01c0b9e3f8635079d6df` with reachable campaign source
`4231b23dc2c46661d01157aac5245bdcf03ce4b1` **only after the compatibility
validation and review below**. Until then the latter is a proposed pin and an
identity-verified campaign executable, not a qualified repository parity oracle
or a demonstrated four-family runtime.

The mandatory latest-peer primary arm is separately resolved at each campaign's
start and frozen through that campaign (ADR-063 F1). Adopting this fixed pin does
not entitle future campaigns to call it latest. Neither pin authorizes replacing
hf2q conversion, quantization or production inference with an external runtime.

## Current source and receipts

- `data/llama_cpp_pin.txt`: old full SHA above. `NOTICE:5–10` identifies the
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
- Today read-only checks confirm `/opt/llama.cpp` still at 970b51ce, clean;
  candidate object is not even in that checkout. The isolated checkout
  `/opt/hf2q-worktrees/adr063-peer-reference` is clean at 4231b23d.
- The archived `adr063-peer-build-receipt.json` records candidate source, binary SHA256
  `98ee8ebf20710ff9b1752b0a4807031b454e927e6aabbaf9e256205927178695`,
  configure/build commands, linked-library hashes and `model_loaded: false`.
  The archived `adr063-peer-version.txt` reports `0.7.0-dev (build 1, commit 4231b23)`
  with AppleClang21/Darwin arm64. Source HEAD and this version alone do not
  qualify parity; retain the receipt's binary/library identities.
- Upstream commit API successfully resolves [4231b23d](https://github.com/ggml-org/llama.cpp/commit/4231b23dc2c46661d01157aac5245bdcf03ce4b1),
  dated 2026-10-23T12:56:59Z, subject `ggml : fix time overflow in ggml_time_us`.
  This establishes reachability today, not runtime/model support.

## Review and change sequence

1. Review this rationale under F1/#356 and record the explicit pin decision.
   Keep the product pin untouched during source/build/compatibility spikes.
   Preserve original receipts and any failed result; do not rewrite old ADR
   benchmark numbers or claim reproduction of an unavailable old executable.
2. Create an isolated hf2q topic worktree from current main and keep the shared
   `/opt/llama.cpp` checkout untouched. Cut all peer tools from the candidate
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
   review, change `data/llama_cpp_pin.txt` to the full candidate SHA. Update the
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
fixtures. These scripts hard-code `/opt/llama.cpp` and output into the checkout;
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
Quantizer/fixture equivalence: not run here. Four-family agentic runtime and
performance qualification: not run here. Proposal does not advance those states.
