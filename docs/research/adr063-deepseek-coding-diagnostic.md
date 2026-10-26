# Explicit DeepSeek HF arm of the actual coding diagnostic

Source-only candidate from main553dc7e26481ea23ce7073d1d4b99f2eac7e1667.
No imports, tests, builds, models, HTTP or GPU execution by source author. This
is a test-client family arm, never production model routing or a cache fallback.
Qwen remains default. The separate unmerged peer adapter is NOT stacked here.

`qwen38_n1_smoke.py --family deepseek4 --diagnostic-slots 1` selects N1;
`qwen_n4_live_smoke.py --family deepseek4` coordinates N4. Historical filenames
stay for reuse. All existing loopback/model/PID/source/compiler/SDK/sandbox/
approval/output/cleanup arguments remain required; no script launches a model.
An externally verified DeepSeek server, canonical source/binary/artifact and
selected resource/reaction packet must be bound before root executes either.

Full PRIMARY workload is unchanged: each actual agent makes seven requests,
reserving8192 outputs EACH, with six real read/read/write/test/read/test tools,
its own fixture/transcript/patch, native assistant reasoning+tool results,
unchanged three strict:false tool schemas and tool_choice:auto, alternating
unary/SSE, overflow reject and complete final-source/test oracles. N4 retains
one server/four independent full262144 contexts and the shared barrier. Existing
short DS scripts with128/768 output caps remain separate smoke controls; this
arm does not substitute those for the primary7×8192 workload.

DeepSeek has distinct approval/result schemas and profiles, explicit family
binding, and preflight keys for native Deepseek4Cache/source controls/resource
packet. It does NOT claim Qwen TQ8, pool24g, repetition1.05 or Qwen2048/512thinking
behavior. Actual selected outer controls must be DeepSeek native cache,262144
per slot,1or4 slots, repetition1.0, required-tool-thinking512, adaptiveprefill,
no graft/persistence/GLP, and explicit request greedy temperature0/max native
reasoning. Greedy sampling is supported by existing DS agentic gate and max
reasoning by structuredtool/canonical coding contract. No explicit thinking
budget/Jinja Qwen kwargs are injected. `max_completion_tokens=8192` remains
unchanged. This is one prospectively fixed diagnostic configuration, not a
claim of engine-best tuning or permission to silently change productive modes.

Approval booleans are not observations: root's outer packet must record actual
/health model/backend/context, runtime SlotAware count/family/config/warmup,
canonical argv/settings and model metadata. Served model ID must be the actual
artifact name, not a filename guess or Qwen label. The client rejects mixed
family approval/schema/result sets and divided contexts through exact typed
controls. It never infers that requesting DeepSeek family loads an architecture.

OpenAI unary/SSE usage and existing strict positive continuation cached-count
checks are shared without normalization: missing/invalid/zero followup counts
fail the existing preliminary reuse predicate. DeepSeek native renderer and
reasoning canonicalization are family-specific; positive counts do NOT prove
all unchanged prior input was reused. Full_prefix_reuse remains PendingPostOracle
and final status ProtocolCompleteAwaitingIndependentPatchReview. A future
family-production renderer/tokenizer replay plus independent patch review is
required before diagnostic/quality qualification. No Qwen generation-cue bound
or stable-anchor predicate is imported into the DS cache path.

The actual six tools are run, not canned messages or assumed successes. Model
may fail an auto tool, emit malformed DSML/OpenAI semantics, exceed wire/context,
return stale tests, omit usage or ignore native reasoning: each is a retained
failure with no retry/truncation. No runtime/model/performance qualification or
PR390 warmup/cancel/growth proof follows solely from this client surface. Its
same generic coding task remains an initial diagnostic, not full ADR063F2.

New focused family helper and six proposed pure controls cover Qwen default
wire/contract immutability, DS actual request additions, exact family/cache/
control typing, rejecting Qwen preflights, mixed/divided contexts and unknown
families. Both existing N1 verdict and N4 coordination suites still apply.
Only AST source syntax/diff hygiene were checked by author; root owns tests
and all actual runs after independent frozen-source review.

Root validation: 69 retained Qwen controls and six DeepSeek-family controls passed;
independent source review accepted the explicit family boundaries. No DeepSeek
model has run under this adapter yet. The greedy/max-reasoning diagnostic lane
is explicit; a captured production-client settings lane remains a separate F2
obligation. Exact family-native prefix replay and real memory/quality evidence
remain pending.
