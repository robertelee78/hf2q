# Production Qwen cache allocation plans

This constructor-only extraction adds checked plans consumed by
`HybridKvCache::allocate_with_profile`, standalone TQ allocation and lazy
`ensure_la_capture`. It changes no native API/dependency, cache policy, context,
slots, speculation, kernel or teacher authentication profile. Source basis:
hf2q017ac49d5c608a86ef448f6695dbb5a39e680d70 / published mlx-native0.15.2.
The source-BF16 teacher remains an independent F32-cache profile with its own
limits; its estimator is not reused as a production plan.

Production alternatives remain U8 TQ codes plus F32 norms, or F32 K/V control.
There is no BF16 cache branch. TQ uses one byte per code and floor(D/256).max(1)
norms per position, independent of quantizer codebook bits. Full-attention roots
remain overwrite allocations, hidden beyond cursor; recurrent/conv roots remain
zeroed, both ping-pong legs allocated. Existing reset remains unchanged.

Initial planning checks products, byte/category sums, metadata cardinality,
conv-channel u32 and slot-rank conversions before Metal allocation. Allocation
traversal consumes descriptor dtype/shape/bytes/initialization, preserving layer
schedule, root order, labels and layer/MTP error contexts. At most one MTP slot is
added when requested; no arbitrary context/layer/metadata ceilings are added.
Zero capacity precedes zero sequence error. Empty schedules remain accepted.
Pure descriptor callbacks and actual Metal calls share the same traversal.

Capture tokens0 leaves existing activity untouched. Every positive request
invalidates activity before old-metadata collection/planning. Old metadata uses
checked shape products and the entire Metal backing extent, including views.
Reuse remains nseq1 existing_count>=request, otherwise exact-count only; nseq4
shrink can replace storage. Keep actions neither normalize nor clear retained
storage. Recurrent then conv allocations install immediately in slot order;
activity is published only after all succeed. Allocation failure preserves prior
successful replacements and leaves activity inactive. Base conv history remains
saturating_sub(1).max(1), capture history saturating_sub(1). Zero-byte capture conv
descriptors are represented until the native allocation fails after a successful
recurrent replacement; kernel1 base behavior is not changed.

Plans expose old backing extents plus new requested payload as a conservative
current-owner upper bound and count references without identity deduplication.
Retained-after is logical current storage, not reclamation credit. Old external,
raw, recorded, command or previously retired owners are outside this seam; wrapper
drop/completion/assignment does not prove physical release. No admission,
reservation enforcement, root lifetime closure or complete B/C/R/G bound is wired.

## Header-bound pure fixtures

The tests retain only scalar values from F1 GGUF headers. Serialized block count
includes nextn, which is subtracted before creating the base layer schedule.
Qwen3.8 artifact SHA cf0eb0103c1ffe7b3702670aa794a669bafbe328b95e892268305dd87c2c30d7:
65blocks/nextn1,16baseFA/48LA,KV4/D256,SSM group16/state128/inner6144,
conv4. Qwen3.6 artifact SHA815a22cab21649948352c0eea6e9be0d9d15606e2024d86e7bdb71fb9df08220:
41blocks/nextn1,10baseFA/30LA,KV2/D256,group16/state128/inner4096,conv4.
No tokenizer, payload, private path or model load is required.

At C262144/nseq4, including the single MTP slot:

| Requested payload | Qwen3.8 | Qwen3.6 |
| --- | ---: | ---: |
| TQ full attention | 37077647360B | 11995709440B |
| LA both legs | 1255145472B | 526909440B |
| Initial TQ total / roots | 38332792832B / 260 | 12522618880B / 164 |
| Fresh four-token capture / roots | 2510290944B / 96 | 1053818880B / 60 |
| F32 control total / roots | 147284033536B / 226 | 47771549696B / 142 |

Qwen3.8 four-to-five-token capture current-owner bound is base+old+new
43980947456B,452 reference upper count. Same four-token request adds zero roots;
nseq4 four-to-three replaces96 roots. These numbers are requested payload, not
physical fit or permission to allocate the full profile in a test.

## Validation status

Root validation passed ten focused pure tests with filter
`production_cache_plan_` (10 passed, none failed or ignored), locked all-target /
all-feature check (29.37s), and locked release build (38.96s). No Metal execution,
model, payload or full-context allocation was performed. The pure tests cover
actual scalar fixture totals and descriptor consumption, layout/initialization,
byte-code/norm floor, F32/no-MTP/single-MTP/empty schedule, allocation fault order
and context, capture reuse/shrink/mixed options/full backing extents, activity
failure/publication, zero conv partial-success, and checked overflow/errors.
Recorder/fault controls exercise allocation orchestration, not synthetic model
atomicity or native final-release behavior. Existing cache and teacher regressions
were not executed in this constructor-only slice; no full-context hardware
allocation is implied. View extent accounting is checked via injected metadata
and the production full-backing getter; no real Metal view was allocated. Host
reservation failure was not fault-injected.

```sh
cargo test --locked --bin hf2q production_cache_plan_ -- --nocapture
```

No speed, quality, physical residency, cap-fit, admission or full Q5 claim.
