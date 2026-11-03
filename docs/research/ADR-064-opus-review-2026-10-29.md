# ADR-064 independent Opus review — 2026-10-29

## Review identity and scope

The operator requested independent review through `claude -p`. Claude Code
2.1.309 was invoked with `--model opus --effort high --safe-mode --tools ''
--no-session-persistence --output-format json`. The first successful receipt
identifies the actual model as **`claude-opus-5-5` (Opus 5.5)**.

The reviewer received inline ADR, research report, spike scripts, source excerpts
and receipts. It had no tools and did not independently execute models, read
the repository or recompute input hashes. This is an independent reasoning
review of supplied evidence, not independent hardware acceptance or product
approval. No implementation/release permission derives from its verdict.

Initial prompt SHA-256:
`8fd00cd1391da32ad92c20000d6c4222a1e703c234b6b758244ac7c78190390d`.
Follow-up prompt SHA-256:
`6ab3cb48caf9d8c053682845a99d58a824a9ffd59586c195b5f48aa351abe18f`.
Raw JSON receipts and exact prompts remain local; generated review payloads
are not repository artifacts.

## Initial verdict and dispositions

Initial verdict: revise before accepting the research program; do not block
cheap gating spikes. The review correctly treated the existing probes as
bounded negative/mechanical evidence, not successful model transfer.

| Finding | Disposition in revised ADR / evidence |
| --- | --- |
| 1. Receiver content differs after rendering | D1 defines `R_receiver(L,C)` and native-prefill target, ledger versus receiver-visible completeness, allowed divergences and replay boundaries |
| 2. Prefix stability and anchor placement | D2/H3 require template-aware placement, surviving-prefix measurement and explicit ADR-063 C1 multi-anchor dependency; new S6 executes source-render controls |
| 3. Session protocol excludes clients | D1 permits principal-scoped content-addressed exact restore; IDs are hints; H3 no longer depends on a custom extension |
| 4. Approximate lineage gaps | D2 restricts initial translation inputs to native-exact and depth at most one; derived checkpoints need a separately qualified same-model arm; full replay alone heals exact state |
| 5. No measured execution entry | H0 adds load/prefill/resource baselines, current admission refusal and redirect criteria before costly derivation |
| 6. Directional asymmetry and text encoder | D4 distinguishes the DeepSeek-to-Qwen hypothesis and requires token-only/no-source ablation |
| 7. Training capacity and quant reference | H0/H4 require compute/access/cost and bit-faithful quant/compressor strategy before data generation/training |
| 8. Post-removal load failure | D5 specifies separately admitted sender reload/restore or recoverable no-model state, with separate recovery timing |
| 9. Principals and isolation | D6 distinguishes unauthenticated single-owner loopback from authenticated shared/LAN scopes; no cross-principal sharing |
| 10. Translator trust | D6 treats artifacts as executable state influence, verifies before install, and assigns signing/derivation/revocation work |
| 11. Latency protocol | H7 defines client-observed semantic output, intent versus server-arrival clocks, drain inclusion, twenty paired runs, uncertainty, context/suffix strata, page-cache conditions and A-B-A totals |
| 12. Stochastic quality | H6 separates deterministic critical controls from paired repeated sampling and a predeclared product-owner noninferiority decision |
| 13. Peak lifetime budget | D5 states stagewise peak formula and candidate between-model/overlapped mapping schedules; feature capture is charged |
| 14. Artifact matrix cost | Consequences enumerate revisions/quantizations/fine-tunes and one-time identity verification |
| 15. Rights and training data | H0 is a written rights prerequisite for generation/training; user sessions excluded absent explicit authorization |
| 16. Missing DeepSeek mechanics control | H4 includes a small/random-weight DeepSeek-V4 architecture control before full pair work |
| 17. Trigger, thrash and timeout policy | D5/H2 require authorized switching, shared-endpoint confirmation, fairness/hysteresis, busy outcomes and qualified keepalives |
| 18. Scope of S3/S4 | Research report preserves synthetic/deepcopy scope; neither is native serialization or model-quality proof |
| 19. Shape-only DeepSeek fingerprint | D3/H1 explicitly bind exact artifact identity rather than trusting geometry/quant labels |
| 20. Client tool execution ownership | D5 limits hf2q behavior and requires content/revision idempotency for external retries |
| 21. Sensitive state storage | D6 specifies owner-only state files, atomicity, retention and honest deletion semantics |

The reviewer hypothesized reasoning stripping would commonly break returns.
S6 narrows that concern: Qwen's current default preserves thinking; the actual
DeepSeek encoder retains it when tools are present. Other measured settings
rewrite the prefix. The revised design does not assume either universal
stability or universal invalidation.

## Follow-up verdict

The second successful receipt again identifies **`claude-opus-5-5`**. Verdict:
**accept as a Proposed research and design program; no blocking architecture
defects remain for this stage.** All 21 initial findings were resolved or
converted into explicit entry/qualification gates. This does not grant product
approval, full-model admission, training execution or publication authority.

The follow-up supplied seven nonblocking clarifications, incorporated afterward:
H0.0 host-admission levers and stop decision; token-only prefill-surrogate
candidacy; native numerical noise-floor controls; separation of client ledger
rewrites from rendering; H2 switch-policy traceability; a predeclared 20-second
load/reclaim screening threshold; and S6's pinned-fixture/tools-present scope.
Those narrow edits followed the review's recommendations and were subsequently
included in the final-source review below. ADR status remains Proposed.

## Final-source and task-extraction review

The continuation ran Claude Code 2.1.309 with the same tool-free Opus command.
Both additional receipts identify **`claude-opus-5-5`**. The final ADR reviewed
has SHA-256
`244070bb39dbf7e18b36bc50b9db7563f8a3d58060f43f34590149492c7b2d66`.
Final-source/extraction prompt SHA-256:
`a8dcda1777bd91dc74e78ba3aaba35c09aee8b7e7aa7f831f6e423b91cb7de6a`.

Verdict: **ADR accepted as a Proposed research/design program; revise the
task extraction before Project creation.** No architecture blocker remained.
The reviewer confirmed all seven earlier clarifications and the boundaries of
the new source-token, installed-client protocol and no-weight Metal evidence.

| Extraction finding | Disposition |
| --- | --- |
| H3 lacked H2 and external C1/#359 dependencies; H7 lacked H2 | Add H2 to H3/H7 and C1/#359 as H3's external predecessor; never take over C1's attempt |
| H5 falsification incorrectly required a release | H5 delivers a reviewed per-direction implement/falsify/redesign decision; H6 also delivers reviewed qualification evidence, not enablement |
| D2/D5/D6 invariants lacked owning verification | Assign provenance/depth fault controls to H3/H6, recovery/single-generation publication and translator trust to H7, and principal/state-file isolation to H2 |
| GitHub cannot resolve an unpublished source commit | Explicitly mark the binding unpublished-local; do not amend/rebase it; no push permission is inferred |

The corrected extraction was reviewed separately against the unchanged full
ADR. Follow-up prompt SHA-256:
`0a1bb1eacb65856c4d3b6e5637d56463c0805f9b0b407934c494bf76ae878591`.
Verdict: **ACCEPT for Project creation and task extraction; all three blocking
findings resolved.** No implementation readiness or product acceptance follows.
Four nonblocking issue-prose clarifications were retained: verify artifacts
before sender eviction; expose the selected path/reason/cost; explicitly check
Rust/`mlx-native` production ownership; and distinguish H8's already completed
ADR review from its H7-dependent future product-release review.

The operator authorized one AWA-managed Project with H0–H8 tasks and a local
topic-branch source commit. Production execution, push, merge, release and
translator publication remain separately authorized operations. Deterministic
source-quote validation against the resulting commit/blob remains mandatory
before issue creation. Raw prompts/results and the extraction manifest remain
outside the repository.

## Unperformed qualification

No full-model native eviction/restore, learned translator fitting, Metal mapper
benchmark, hf2q-backed live OMP/OpenCode inference A-B-A, repeated-switch coding qualification
or end-to-end switch SLO is claimed. H0 names these outstanding evidence gates;
the core feature remains the deliverable rather than a literal paper port.
