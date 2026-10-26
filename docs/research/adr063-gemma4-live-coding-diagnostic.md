# Gemma4 family-explicit live coding diagnostic source contract

Source-only candidate, 2026-10-26, base91e3bf1b. No author imports, tests, builds,
HTTP, tools, model/device work or executed readiness proof. Root may execute only
after the active DeepSeek server unloads. One hf2q instance, never a peer fallback.
This is diagnostic infrastructure, not admission, speed or quality acceptance.

Use the existing client with `--family gemma4 --engine-kind hf2q`; N1 retains
seven real requests, six actual prescribed tools, max8192 per request, all tool
results and alternating unary/SSE. N4 uses the existing four-client barrier and
one server with four independent context262144 slots. No forced tool choice,
scripted assistant, retry, truncation or output/context reduction. Existing
sandbox/process/tool code is unchanged. Unsupported Gemma peer is rejected.

Family profile/schema is `adr063-gemma4-n{1|4}-live-v1` and corresponding
`hf2q.gemma4-n1-client-*`/`hf2q.gemma4-n4-*`. Explicit Gemma preflight and
family_controls replace Qwen AUTO/TQ/pool and DS native-reasoning labels.
Requests preserve captured omissions (including temperature), toolAuto,
strictfalse, logprobsfalse, no response_format, overflow reject. Gemma adds NO
DeepSeek reasoning_effort=max or Qwen thinking budget. Sampling defaults must be
captured from actual readiness; omission is not a peer-default claim.

## Operator/source binding

Canonical operator entrypoint is `scripts/serve_gemma4_opencode.sh`. It uses
inflight-batched, maxslots4 default, sharedKV48GiB, repetition1.05,
HF2Q_KV_LCP_RESUME_CAPACITY8g, BATCHED unset/auto and optional existing mmproj.
Do not claim no projector merely because requests have no images. Launcher
MODEL default is a different APEX Q5 artifact: this diagnostic MUST explicitly
bind the selected hf2q-produced Google Gemma4 Q4_K_M artifact, declared F1SHA
bf65b7e163216885790442d0324839101ba4b5b49f7be4ec0cf3d7c5c538ef53,
16796018144bytes. Local model path is supplied privately; no weights are read
by this adapter. Context262144 must be explicit/verified against readiness;
launcher doesn't itself supply a --ctx override. Root owns binary/pin/launch proof.

Actual loader `engine.rs::GemmaLoadedModel` validates selected GGUF embedded
Gemma turn/tool markers; the canonical launcher's APEX description must NOT be
substituted for this artifact's real embedded template. Bind runtime tokenizer/
template semantic fingerprints and exact metadata export. Do not use Qwen chat
rendering or DS native encoder. `INVESTIGATION_ENV.hybrid_kv` defaults true,
codebook bits8; `decode_batch_gemma4` selects the provisioned Hybrid regime and
errors if requested Hybrid absent, not silently FullTq. FullTq is an explicit
alternate requiring a distinct packet; current adapter refuses it. HB scaffold
may coexist with Hybrid physically, so the label is not a storage dedup credit.

Selected controls require cache_family gemma4, slots1or4, context262144,
hybrid_kvtrue, codebook8, LCP8GiB, repetition1.05, prefill_routeauto, graftfalse,
disk_persistencefalse, glpfalse, external_drafterfalse. These are prospective
packet bindings, not live observations invented by the client. Gemma's existing
batched target body/heads stay in use; no Qwen AUTO/MTP switch or DS speculation
policy is imposed. Canonical script supplies no external drafter; root must
verify actual request/default graph and absence of unbound speculative overrides.
No useful default is disabled to produce a favorable result.

## Prefix and execution evidence still required

Client protocol success remains awaiting independent patch review and post-oracle.
`engine.rs` Gemma stable-boundary/prefill progress uses a nonempty boundary inside
prompt length and Hybrid sliding-state anchors; normal follow-ups process only
uncached suffixes. Cached_tokens>0 alone is insufficient. A future Gemma offline
oracle must call the selected artifact's actual production render/tokenize and
thinking kwargs, derive its true stable render boundary and require prior stable
IDs prefix current prompt plus cached>=prior stable length, exact usage/cold0.
Do not reuse DeepSeek P[:-8] or assume Qwen generation-cue semantics without the
Gemma renderer proof. No oracle implementation or actual full-prefix claim is
included in this adapter.

Root freezes resource/pressure/cleanup and exact source/client hashes before any
launch. Gemma persistent duplicate scaffolds, LCP old+new owners and live queued
scratch remain its own Q5 ledger. This adapter adds no memory cap/controller,
quantization, inference graph, production routing, native kernel or policy change.
Source controls exercise actual builder→wire→exchange archive boundary (mocked
connection sentinel prevents HTTP), explicit control rejection and peer refusal;
author has not executed them. Existing Qwen/DeepSeek defaults remain unchanged.
