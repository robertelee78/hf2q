# Optional Qwen pool creation profile

Source implementation with root-owned narrow validation completed; source author
performed no builds, tests, device or model execution. Native55 is available from
public registry mlx-native0.15.3. This candidate was validated on the verified
0.15.4 pin from PR388; this slice adds no native change or private path patch.

The actual central registry accepts only
`HF2Q_QWEN_POOL_GROWTH_PROFILE=adr063-24g-v1` with
`HF2Q_UNSAFE_EXPERIMENTS=1`; unset remains legacy. Invalid/unacknowledged intent
cannot silently serve uncapped. The actual Qwen SlotAware spawn installs one
process authority before worker creation, then registers the real worker before
Warmup/request acceptance. A cache replacement is refused before creating another device/queue/registry.
The verifier's actual GPU_CACHE device binds the
empty facade before scratch use; the KV provisioning device is not substituted.
Every selected handout and warm reuse checks native thread/device/queue/residency
identity. A global first-use/install atomic prevents prior legacy history on
another thread from being hidden by a fresh TLS instance.

The immutable policy is24GiB/16,384 roots/32,768 attempts, one pool/generation,
rank8/batch1 and powers2 through16GiB. Per-bucket root ceiling is
min(16384,floor(Q/b)); all process/pool/bucket limits apply simultaneously.
This is an enforcing policy, **not fit or admission**. Preallocated free-list
handle slots total368,638; metadata/allocator bytes are additional. No successful
creation refunds on reset/clear/drop. Quota/identity/reset failures poison the
profile before returning, surviving stringification. Real AUTO prompt-MTP
catch/replay consults the sticky state; unknown forward/context errors and worker
GPU-fatal paths quarantine rather than clearing/resetting the state to retry.

Existing K grouping, barriers, waits, TQ and AUTO remain unchanged. A reset
requires BOTH no future accesses through pool-issued roots and successful GPU
drain. Direct FA/DN/FFN/chunk arenas, positions, layer-boundary/rings and durable
nextn parents are not in the resetting pool. Pooled DN attention output is
consumed by that layer's FFN before K reset; next-layer hidden is a direct ring
root. Hold clones after their last access only drop. Intra-K and held-terminal
paths do not reset. Greedy pooled positions persist through the whole call and
are reused only after it returns. Residency strong retention alone never
protects an issued root from recycling. The unsafe native ordered-reset API is
not used for this submitted EncoderSession lineage.

Selected text/tool eligibility refuses unsupported capture/vision/embedding and
MTP_PROFILE and split-commit diagnostic routes explicitly. It does not disable canonical projector
loading, speculation, TQ or useful defaults to manufacture fit. All corresponding
direct/loaded ownership still belongs in the complete memory packet.

Pure tests exercise the real parser/policy/atomic-install/failure/eligibility
seams without mutating global authority or loading Metal. Proposed root filter:
`cargo test --locked --bin hf2q qwen_pool_ -- --test-threads=1`.
The following ignored controls require independent packet review and separate
fresh processes with EXACT selected test names; root executed each once after
review, with results recorded below:

- `qwen_pool_tiny_native_quota_reuse_and_queue_identity`: public Native55
  mechanism192B/two roots; wrong new queue refuses without changing accounting,
  cloned actual device reuses, quota refusal preserves successful counts. No GPU.
- `qwen_pool_tiny_actual_hf_route_sticky_failure`: actual HF facade/shared
  installation boundary with private test-only192B policy, exact profile+ack
  required; warm reuse then quota refusal poisons/reset/replay refusal. No GPU.
- `qwen_pool_tiny_drained_windows_preserve_direct_parent`: actual HF facade with
  private64B test quota and profile+ack+HF2Q_ENCODER_SESSION=1; four direct64B roots
  plus one pooled64B root=320B, no model. Two synthetic windows/four native
  add/mul dispatches/two drained submissions, independent exact CPU formulas,
  signed input, sentinel outputs; a direct returned row-view's full parent and
  future-used CPU-filled direct input survive reset/reuse unchanged.
- `qwen_pool_tiny_legacy_drained_windows_reference`: SAME window-control body
  with profile unset/session mode; matched CPU oracle, no authority installed.

Each actual case must produce its positive complete marker and one passed test;
no device skip/zero-filter qualifies. Never invoke all ignored controls in one
process: Native authority is intentionally process-once. The tiny policies use
the same private install/bind/routing boundary, not runtime profile-size knobs.
Requested payload excludes metadata/driver/session setup. Both GPU windows encode
all fallible work before each submit; previous window is fully drained before
next encode; wait precedes CPU reads. No forbidden escaped-pooled-owner race is
executed. These are mechanism/alias controls, not logits/model/correctness/speed
or a proof that24GiB is viable for either selected model.

Q38 W+C+capture-transition plus24GiB pool already totals85,923,004,416 requested
bytes before direct/CPU/G; a host with about79GiB currently free may fail. No run
or lower cap is automatically attempted. Known direct prefix bridges, IdMm/MTP,
CPU anchors/logits/queues/persistence and allocator/driver retirement G remain
separate, as do C/R and family/Metal policy limits. Full Q5 and all-four model
speed/admission claims remain open; actual model activation requires a separately
reviewed complete packet and external supervision.

Non-serve teacher/tree entry points are not granted this profile by analogy: the
actual installer admits only the selected serve worker; unregistered entries
refuse before its pooled route. This is not a universal guard for all direct
allocators. Unknown submitted-work failure remains the existing fatal/OS-recovery
branch; the profile neither adds a GPU wait nor claims successful-drain release
credit on that branch. No actual model activation is authorized here.

Direct infallible `Engine::spawn` rejects requested/invalid profile intent with a
clear documented precondition panic BEFORE channel/metadata/thread creation.
Use fallible `spawn_with_mode` SlotAware, which returns typed startup refusal.
The public signature/default legacy contract is unchanged; no poisoned dummy
engine is created. A pure actual-precondition test covers refusal without any
LoadedModel, channel or device construction.

## Retained root validation

On the exact published native0.15.4 dependency, locked all-target/all-feature
check passed in27.44s, release build in35.61s and test-binary compilation in49.53s.
The actual qwen_pool_ filter ran13pure tests successfully and left4hardware
controls ignored. Five supervisor qualification controls passed separately.
All four separately supervised fresh-process controls then passed their exact
positive markers and one-test harness result, including the192B HF sticky-failure
route and both320B synthetic two-window GPU/CPU-oracle arms. Source and binary
hashes remained bound throughout. Per-case normal-pressure and free-memory guard
records are retained. These results do not exercise a full model or allocate the
24GiB policy ceiling. Independent final evidence review and CI remain integration
gates; complete B/C/R/G, full-model tool/SSE/quality and all-four wins remain open.
