# Shared-target real cancellation lifecycle extension (source only)

This extends scripts/test_agentic_cache_lifecycle.sh under an explicit opt-in;
the existing default lifecycle continues unchanged. New wrapper
scripts/test_qwen_shared_cancellation.sh requires a fresh absolute OUT_DIR,
actual SERVER_LOG and SHARED_LIFECYCLE_EXCLUSIVE=1. Root must establish that
this is a dedicated already-running Qwen38 candidate with no external clients,
shared experiment enabled, normal AUTO/ON24 and exact source/binary/launch
bindings. This flag is a prerequisite declaration, NOT an independent proof.
Root supervisor owns model/process/resource admission and eventual cleanup;
this harness never starts or stops a server. No new model/prompt download.

The base tool-call and real tool-result continuation are existing lifecycle
steps. Default128 generated local context lines keep this fixture small; no
large prompt is needed to manufacture overlap. Active request uses retained
prefix and unchanged tool surface. After its first semantic event, launch an
independent streaming survivor (different system/user conversation, same tools,
normal temperature0/default policy; no forced SpecOFF or controller mutation).
Before submitting the queued exact retry, require an ACTUAL new shared census
line with width2, two distinct scheduler request IDs/slots/generations and the
128singleton-attention/66shared-operator maps. Only active and survivor are in
flight during this witness phase. Old log lines are excluded by a fresh byte
offset after base+seed completion. Actual inode/truncation/count/geometry are
checked; raw line, offsets, observed monotonic time and curl PIDs are retained.

No shared line within180s, completed member, or missing distinct cohort exits
INCONCLUSIVE3; it cannot pass from HTTP overlap or an old census. Malformed
source evidence fails. The exclusive dedicated-server/raw log prerequisite
must be reviewed externally to join the two scheduler IDs to these two client
requests: the current census does NOT export HTTP request UUIDs. Do not claim
that a single log line independently proves this external exclusivity.

Then queue the existing exact retry and recheck both connections/nonterminal
streams immediately before client transport cancellation. Record cancellation
boundary, close active curl, require no terminal DONE from canceled stream,
require survivor valid SSE content/usage/DONE and exact-retry valid unary
continuation with high cached-prefix reuse. Existing isolation request must
return exactlyISOLATION_OK with template-sized-only cache reuse, and leak none
of active or survivor markers. Same tools/rendered prefix are retained across
active/retry. All raw requests/responses remain available for root's exact
post-unload token-vector helper if a stronger full-prefix oracle is needed;
this lifecycle gate does not promote counts into that separate proof.

Boundedness: six HTTP requests total (base,seed,active,survivor,retry,isolation),
existing per-request connect5s/deadline900s and output-token limits (active and
survivor2048) remain; live-census wait180s, log scan4MiB total, each stream16MiB
cap before cancellation. Root can set stricter existing CURL_MAX_TIME_SECONDS/
SEMANTIC_WAIT_SECONDS and its whole-case supervisor; those are prospective
operating settings, not model-quality changes. Client EXIT/INT/TERM cleanup
terminates/reaps all three owned curl children. No parallel model instance.

This tests real client-close handling after a witnessed shared healthy target,
not cancellation mid-Metal CB, injected GPU failure, EOS correctness, full-body
B4 or S3/performance acceptance. Host controls call actual census parser for
valid requestID0, duplicate owners, missing maps/wrong width, geometry and mere
HTTP overlap rejection. All authored code and five controls are UNRUN by the
author. Root owns execution after publication/parity and independent review.
