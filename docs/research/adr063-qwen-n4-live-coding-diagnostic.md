# Prospective four-agent live coding diagnostic

Source-only infrastructure. One external hf2q server, MAX_SLOTS4 and full
context262144 EACH, serves four independent Python clients. Each completes seven
real generations, six actual tools and a final summary: total28requests/24tools,
8192 output allowance per generation. This is a real N4 coding diagnostic, not a
matched speed claim, full-memory proof or actual qualification result.

No server/model starts in these scripts. The coordinator starts exactly four
Python clients; hf2q binary/model ownership, readiness and final server cleanup
remain with the root-owned external wrapper. Never start a second hf2q instance.
Logical slots4 do not change Native creation max_batch1: that is a separate native
allocation contract. No production Rust/native/pool defaults are changed here.

## Reuse and workload

The reviewed N1 Tools, process/sandbox helper, protocol verdict and53-control
harness are reused byte-for-byte from the frozen N1 source tree. N1 archives are
not modified. The client supports defaultN1 or explicit --diagnostic-slots4;
N4 schema/profile/preflight fields honestly require slots4, never relabel slots1.
Tool package/internal fixture schema names remain N1-named reusable mechanisms;
that does not claim one slot on the shared server.

New profile `adr063-qwen-n4-live-v1`, workload
`adr063-qwen-n4-parse-port-v1`. Initial system text adds a distinct agent-0…agent-3
identity. The prompt uses stable workload/agent identity, NOT per-attempt run IDs,
so future arms can freeze the same prompt fingerprints. Run IDs and random
barrier nonce bind the local attempt and receipts separately. Original canonical
system/user task, three schemas, initial explicit six-step workflow and effective
sampling are unchanged otherwise. No retrospective instructions/scriptedassistant
replay/forcedtoolchoice/retry/truncation/summary. Complete actual reasoning/tool
calls/results are retained independently for each client.

Every client uses auto, strictfalse, logprobsfalse, no response_format and full8192.
Captured sampling omissions resolve greedy0/topP1/topK0/minP0/seedabsent, repetition
1.05 and thinking2048/512. Auto lazy tool grammar remains active as production
selects it. Read/write/run_tests actually operate on each agent's distinct owned
Rust fixture; targets ports and all are real. Unary/SSE alternate within each
conversation. Existing wire/source/tool-output and deadline failure envelopes
are preserved, not weakened to make four agents pass.

## Shared start barrier and aggregation

After approval/source/proof validation and private fixture setup, each client
atomically publishes its ready record, identity hash and complete initial-message
hash. No HTTP generation occurs until all four are ready and coordinator
atomically publishes one release bound to all four ready-file hashes, same server
PID/run/nonce. Missing/stale/duplicate identities, same output/prompt, changed
server or missing agent reject. Ready/approval/output fingerprints are checked;
each client verifies its own ready hash and full four-agent release grammar.
Barrier timeout120s; no extra per-turn barriers modify ordinary client behavior.

The coordinator monitors independent client process groups and observed owned
descendants, permits only one bound server PID and aborts on any failed client.
Bounded cleanup uses the reviewed process helper: metrics failures cannot bypass
owned-group TERM/KILL/reap; uncertainty remains failure. No partial3/4 success,
missing results, pending status, changed identity, nonzero exit or unknown cleanup
can qualify. Whole coordinator deadline7320s includes120s barrier plus retained
7200s per-client run allowance; request900s and tool45s limits are unchanged.

Successful exit0 means ONLY
`FourAgentProtocolCompleteAwaitingIndependentReview`. Each child must have
`ProtocolCompleteAwaitingIndependentPatchReview`,7turns/6tools and cleanup.
Every agent's raw patch/tests/toolsemantics and exact prefix replay still need
independent review. `full_prefix_reuse=PendingPostOracle`, runtime/model_qualified
false. Positive cached counts alone are not full-prefix proof. No observed N4
result, model correctness, tool-quality or speed is invented by source existence.

## Root wrapper interface

Root owns exactly one canonical Qwen38 server launch (MODEL vanilla, MAX_SLOTS4,
C262144, AUTO/TQ8, selectedpool+ack). The launcher was set QWEN38_VISION=off,
but the core autoloaded a projector; the actual requests were image-free. This
is NOT evidence that no projector was loaded or resident. Record its actual
identity/residency and keep image-free workload distinct from loaded capability.
No GLP/graft/disk persistence is selected. Root owns actual listener/PID/binary/model/registry/source
fingerprints and effective configuration evidence. Readiness must prove four
slots at full per-slot context, not ctx divided by slots. This lane is on the
operator's dedicated host with root's live pressure/available-memory policy;
there is no newly invented N1 numeric memory floor or CPU allowance in the N4
client approval. Do not infer fit from the24GiB pool quota or use extra HF binaries.

After readiness, root passes a hash-bound coordinator approval and the unchanged
actual model-free sandbox proof for the exact reused process helper. Tool/sandbox
semantics remain reviewed; no unsandboxed fallback. The wrapper supervises
coordinator/client/tool descendants, live host memory/pressure and server logs,
then unloads/reaps owned server/listener. It preserves all failures and does not
kill unrelated processes or retry with reduced work.

```sh
python3 -E -s -B scripts/four_agent/test_qwen38_n1_verdict.py -v
python3 -E -s -B scripts/four_agent/test_qwen_n4_live_smoke.py -v
python3 -E -s -B scripts/four_agent/qwen_n4_live_smoke.py \
  --base-url http://127.0.0.1:PORT/v1 --model EXACT --server-pid PID \
  --output /private/new-run/n4 --cargo /verified/toolchain/bin/cargo \
  --toolchain EXACT --sandbox-exec /usr/bin/sandbox-exec \
  --sandbox-proof /private/proof.json --sandbox-proof-sha256 SHA256 \
  --approval /private/n4-approval.json --approval-sha256 SHA256
```

Coordinator output must be new/outside Git. It creates agent-0…agent-3 child roots,
complete per-agent command/approval/raw/transcript/tool/process records, shared
barrier records and aggregate summary. Child stdout/stderr remain per-agent.
Copies do not share fixture paths, test targets, prompts, mutable transcripts or
HTTP connections. Root wrapper need only replace its N1 client invocation with
this coordinator and MAX_SLOTS4 readiness; it must not start four servers.

Coordinator approval schema `hf2q.qwen-n4-coordinator-approval.v1` fields:
profile, approvedtrue, base_url/model/server_pid, run_id (16–64 safe chars),
barrier_token (32 lowercase hex), python_sha256, sources mapping exactly MODULES
(seven Python basenames/hash), and agent_template. agent_template carries the
existing source-bound cargo/sandbox/proof/SDK/toolchain/fixture/base-workflow
identities, approved flags and preflight evidence. Preflight is the existing N1
set except slots1 replaced by `ctx262144_slots4_verified`, and old N1 CPU numeric
cell replaced by `dedicated_host_live_pressure_availablememory`. Other actual
source/model/profile/AUTO/TQ/grammar/guard/sandbox facts remain verified. The
legacy-named auto_tq8_text_only_verified key certifies image-free requests in this
record; it does NOT prove absent projector. Actual autoloaded projector disposition
must accompany the run receipt rather than being inferred from that key.

Coordinator writes `hf2q.qwen-n4-agent-approval.v1` for each child, binding slots4,
agent/run/one-server PID, output, barrier path/nonce and initial identity SHA.
Children independently revalidate every approved module hash/proof/control before
ready. Child result schema `hf2q.qwen-n4-agent-result.v1`; coordinator result
`hf2q.qwen-n4-coordinator-result.v1`. These honest N4 records do not claim slots1
readiness or a new sandbox proof. Private approval/source manifest/root raw proof
must be independently reviewed before any future execution.

## Validation status

Author performed source writes/reads/hashes only. Retained53 existing controls
and new focused N4 controls have NOT been imported or executed here; no tool,
compiler, sandbox, HTTP, model/device operation or additional HF process ran.
No source commit/build/launch authorization is conferred by this note. Future
actual results must record exact source/binary/settings,28requests,24tools,
per-agent semantic TTFT/cache evidence and all cleanup, without partial scoring.


Documentation correction: actual core projector autoload is recorded above; frozen
Python and prior raw receipts are unchanged. Launcher vision-off intent alone did
not prove absence. This correction supplies no new comparative or speed result.

## Prospective same-binary pool ON/OFF ablation

N4 approvals now replace the inherited N1 `profile24g_ack_verified` with
`pool_profile_selection_verified`. This is a verified explicit selection, not
a claim24GiB quota is enabled for OFF. The N1 legacy approval contract is unchanged.
Archived successful N4 records retain their original source/key and remain valid
for that recorded ON run; do not rewrite or rescore them.

Both future arms must use the same exact hf2q binary/source/registry/model and
this same newly frozen client/workload. ON explicitly sets
HF2Q_QWEN_POOL_GROWTH_PROFILE=adr063-24g-v1 and HF2Q_UNSAFE_EXPERIMENTS=1. OFF
removes BOTH keys from the full launch environment, including inherited values;
setting the profile string to `off` is not the supported Off parser contract.
Record actual selection/environment and source-inferred binding honestly; no
observed native quota counter is invented. The runtime model pool is distinct
from the scratch allocation pool, so retain its existing one-model/slots checks.

This is a narrow ON/OFF allocation-policy ablation, not the mandatory stock-release
or latest-peer comparison and not an all-model speed verdict. No contexts, slots,
AUTO/TQ/thinking/tools/projector/autoload or cleanup/watchdog policy changes accompany
the switch. Dedicated-host live pressure/available-memory supervision remains
root ownership. Any failed arm stays failed, with no reduced-work retry.
