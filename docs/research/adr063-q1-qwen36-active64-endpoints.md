# Q1 Qwen3.6 full active64 ordinary endpoint evidence

This source-only extension uses the existing explicitly selected Qwen3.6 MoE
model implementation (`qwen35moe`, `Qwen35Variant::Moe`), not an approximate
Qwen3.8 loader or another family's state. It completes the missing ordinary
T1 before/restored64 and T3 checked64 endpoint captures. No real-model result
or whole Q1 qualification is claimed.

The existing real forward_gpu All call still returns every row:128+16 for cold,
64+64+16 for T1/T3, four distinct frozen144-token logical lanes, independently
owned single-slot caches. T1 already takes a real snapshot_prefix64, constructs
a new actual cache and calls restore_partial64. A new typed observer event at
that exact synchronous restore return captures full active64 before replay;
it emits no fabricated logits or GPU submit ID and adds no GPU wait/math.
The prior checked forward source and synchronous CPU-copy restore provenance
are recorded separately. Current ping-pong getters and restored canonical
parity are observed rather than assumed from named buffer fields.

The selected real allocator descriptors generate every active target full-attn
head's TQ8 packed K/V and norms and every DeltaNet conv/recurrent state, with
actual MTP cursor0 for this ordinary target-only route. Before64 has ordinary
segments; restored64 uses segment+1000. Expected manifests require all four
lanes and every role; capture checks actual cursor/shape/scalar/visible extent.
A complete paired full-content hash, shape, scalar and byte-count gate runs
after Store writes all evidence. Its failure is retained in execution.json
before the actual test returns failure. Cursor-only equality cannot pass.
Packed U8 is encoded floating cache state; exact restore bytes are a copy
contract, not a discrete expert-ID invariant or a decoded-value tolerance.

Header-derived complete cache roles plus all576 full-vocabulary F32 logit rows
are checked with overflow-safe arithmetic against explicit positive payload
quota<=2GiB before model load. Store independently rechecks the same plan;
loaded config must reproduce it. Controls invoke the actual planner for the
frozen Qwen3.6 shape, require manifest/packet bounds, check exact-quota-minus1
and overflow rejection, and exercise the actual hash gate against changed,
missing, duplicated and layout-resealed records. These controls are authored,
not executed. Payload is captured output I/O, not whole host/process fit.

Runtime resources still include model weights/activation/cache ownership,
old cache plus newly allocated replacement plus snapshot during T1 construction,
per-call full logit download and temporary encoding, metadata, persisted
artifacts and driver/retirement allowances. No old cache or file-backed memory
is credited as physically freed. A separately reviewed finite supervisor and
actual source/bin/model/workload acquisition remain mandatory before execution.

The new64 role requirement applies only to explicit Qwen3.6 ordinary T1/T3;
Qwen3.8's existing endpoint manifest and route remain unchanged. T0 has no
invented64 checkpoint. T2 remains the source-grounded ordinary serial-handle
capability refusal, not manufactured physicalB4. T6 still requires the separate
real owning-worker adapter/full144 extension; T5 still requires its actual
proposal adapter/coverage. Routing-margin/native witness integration remains
pending a published dependency and exact HF pin; no CPU resoftmax proxy is
introduced. Cross-trajectory floats stay Unspecified/Inconclusive, even if
restored64 encoded bytes match. No performance claim is made.
