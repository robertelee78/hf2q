# ADR-063 Q5: selected HF profile operating ledgers

**Draft, 2026-10-27; no Q5 acceptance or new execution authority.** This joins
existing source-derived payload cells and prospectively declared operating
allowances. Qwen3.6 and Gemma have numeric operating packets below; Qwen3.8 and
DeepSeek still have specifically identified source joins. An executed normal
conversation or an available-RAM threshold does not fill those joins.

Governing [ADR-063 Q5](../adr/ADR-063-apple-silicon-four-agent-speed-coherence.md)
and the [qualification matrix](adr063-q5-final-qualification-matrix.md) retain
stock, experiment and eligible-peer obligations. This document covers HF only;
none of its kernel/cache/pool quantities is transferable to a peer engine.

## Units, policy and identities

All arithmetic uses integer bytes; GiB=1073741824. The selected host has
137438953472 bytes. Each profile retains four independent262144-position slots,
8192 output allowance, text-only requests, native family template/tokenizer and
its supported modes. Largest source-admitted Qwen input is253888 with graft0:
P+8192+64<=262144. Gemma requires P+8192<=262144, giving253952. Do not silently
remove Qwen's64-row safety reserve or reduce output/context to fit.

The operator's dedicated-host available-RAM policy supersedes earlier generic
12GiB reserve and DS116GiB operating assumptions. The pure
[available-RAM ledger](adr063-q5-ledger-slice1-2026-10-23.md) distinguishes an
explicit additional family limit from None/Unknown. Metal recommended working
set is a performance recommendation; it is not a new hard physical ceiling.
Retained historical rejections under older policies remain historical evidence.

A packet evaluates one phase maximum with old/new coexistence:

`B_operating = max(load, warmup, ready, prefill, decode, growth, publication)
              + separately named CPU/tool/client allowances + positive G/J`.

A conservative sum may dominate several exclusive phases; label that overbound
rather than call it measured simultaneous residency. Do not add a subtotal and
its component rows. Shared mappings/views/Arc/raw command bindings retain a
parent, but do not create another payload. Completion, reset, clear and logical
Drop do not prove final physical retirement. Known large arrays never belong in
opaque G. G covers metadata/allocator/driver and bounded retirement; J covers
observation age, stop latency and in-flight overshoot once.

Free+CachedFiles is the recorded **operating availability metric**, not a generic
cache refund or a disjoint physical ownership certificate. In this formulation
do not subtract the same observed competition again. Fresh aggregate competitor
and headroom evidence, pressure, process identity/footprint and cleanup remain
required. An absent physical high-water or maximum stop overshoot stays
Unavailable, not zero. Numeric G/J below are declared positive allowances,
not measured universal maxima or hard OS enforcement.

| Family/profile | Source/evidence identity | Public native | Artifact SHA256 |
|---|---|---|---|
| Q36 stock | retained runtime2d7599907be92d17736bc205806907589e845306; metadataa3fcbdb0 and corrected source admission are separate helper identities |0.15.5|815a22cab21649948352c0eea6e9be0d9d15606e2024d86e7bdb71fb9df08220|
| Q36 optional bounded pool | same family graph, explicit adr063-24g-v1 startup selection required; not selected in historical173831 stock run |0.15.5 in retained2d lock; new selection must be frozen|same Q36 artifact|
| Q38 bounded pool ON/OFF evidence |5a9d5e9f3e895151e1aab23ddbf2277a1e559253, fixed20cohort campaign070338 |0.15.4, cffc3bb53d4823ab39879ddeb45ee16b9d902478|cf0eb0103c1ffe7b3702670aa794a669bafbe328b95e892268305dd87c2c30d7|
| Gemma checkpoint study |18162bb702806ac6c083bf01ec80f4675207fda4 production; exact study helper/source composition documented separately |0.15.5|bf65b7e163216885790442d0324839101ba4b5b49f7be4ec0cf3d7c5c538ef53|
| DS bootstrap/grammar candidate |2d759990 bootstrap/cancel/growth evidence;56704f29783e011ab2c2a57c9eaeb80df3d68050 grammar N4 evidence |0.15.5 /0.15.6 respectively|50df7e3dfe06321613c78075df4111b14985a0db31087992d7438f2895a3aad6|

Current published native0.15.6 capability does not relabel older0154/0155 runs.
Optional native live requested-extents enforcement is not automatically installed
by these HF profiles and is not their whole-process physical admission proof.
One model prime, cache provisioning and serving-worker generation are bound;
replacement/reprime is not free. Automatic projector loading must be read from
actual runtime metadata, not inferred absent from image-free HTTP traffic.

## Qwen3.6: numeric stock and optional-pool packets

Source geometry: qwen35moe,40layers(30linear/10full), oneMTP, h2048,
FA16/KV2/D256, linearK16/V32/K128/V128, E256/top8/I512/shared512.
AUTO depth<=3, TQ8, encoder session1, ordinary K8; full40-layer short verifier
has<=4rows. No DWQ/GLP/graft/disk persistence/projector; actual173831 runtime
reported projectorfalse. One worker's transaction arena is not multiplied by4.

| Disjoint named domain / conservative transition | Bytes | Alias/coexistence rule |
|---|---:|---|
| Retained GPU weights plus CPU weight arrays |22426621440|Includes587584512 CPU router/shared values; no second whole-file mmap charge|
| One four-sequence TQ/FA/MTP/recurrent cache |12522618880|Both recurrent/conv legs included; no fifth Qwen bootstrap|
| Capture old+new maximum |2107637760|Replaces, not adds to, single capture1053818880|
| Stock legacy pooled-root history |10737418240|Rounded10GiB source bound, not enforced quota; includes free+issued union, not imported labels again|
| Direct outer arenas and embedding |1215963436|One R2048 transaction: FA58720256, projections201326612, DN370147420, MoE249700364, chunk134479872, nativechunk117702832, outputring33554432, boundary33554432, embedding16777216|
| Full-prefix direct bridge |1694498816|F32dequant plus BF16K/V and query/output; one layer's checked terminal join, not40copies|
| Separate IdMm TLS old+new |100669440|Three gate/up/down scratch pairs, outside legacy pool; replacement overlap included|
| Durable hidden parents |218103808|13 full R2048 parents; last-row views do not shrink backing|
| Five CPU anchors |329318400|Four retained plus construction; explicit payload, outside CPU-other allowance below|
| Direct MTP stage |536870912|Long catch-up KV-only; two pooled K-norm/RoPE roots already charged in pool, no R*vocab head|
| **Named subtotal** |**51889721132**|**48.326068681GiB**|

Stock pool derivation bounds each role's retained power-of-two bucket union by
2*m*B. Separate largeR<=4096/K8 and verifierR<=4/K40 epochs; scalar legacy
attention+MoEFFN padded64handouts/layer,40base+4MTP slots; each scalar bucket<=1MiB.
Surviving large pooled DN output and FA SDPA/vector scratch, head and MTP KV
roles yield9609681408B, rounded upward to10GiB. R>1MoE/DN/FA projection arenas
exclude their large legacy scratch callsites. Selected encoded Q4_K/Q6_K and
pre-uploaded BF16 router/shared weights exclude floating-weight conversion.
Source seams: qwen35/decode_pool.rs, forward_gpu.rs4044/4250–4448/6168,
gpu_delta_net.rs4166/4348, gpu_full_attn.rs1499/5207/5400/6778 and
mtp.rs453–580/631/643. Native buffer_pool.rs keys exact power-of-two byte bins.
Changing reset defaults, catalogue, worker count or independent draft-loop
cardinality reopens this source bound. Failure/quarantine cannot take healthy
reset credit. The512MiB MTP direct envelope retains the known~302MB arrays;
native quantized projection/TQ append selected dispatches write supplied outputs,
not a hidden full-weight expansion.

| Additional operating cell | Bytes | Scope |
|---|---:|---|
| Server CPU/owned I/O other than explicit weights/anchors |3221225472|Tokenizer/HTTP/grammar/log/transcript/container allowance|
| Tools/fixture I/O |3221225472|2GiB process-tree allowance plus1GiB aggregate fresh-target allowance|
| Client interpreter/control |134217728|Not a second transcript charge|
| G: driver/rounding/retirement |2147483648|Positive operating assumption, not measured final-owner release|
| Observation age |2147483648|J part1|
| Runtime unavailable headroom |3221225472|Reserve2GiB plus reaction/overshoot1GiB, J part2; not charged again|
| **Additional subtotal** |**14092861440**|**13.125GiB**|
| **Stock explained operating packet** |**65982582572**|**61.451068681GiB**|
| **Optional24GiB pool explained packet** |**81014968108**|**75.451068681GiB**; replaces10GiB pool, never adds both|

Arithmetic correction: the prior private numeric review printed the named subtotal
as51889719132B,2000B below the sum of its listed rows. The GiB figures were
already correct. This draft uses the exact sum51889721132B and operating total
65982582572B; optional24GiB becomes81014968108B. Existing hash-bound live
receipts are unchanged. This2000B transcription correction changes no source
allocation, positive allowance,96GiB policy or historical outcome.

The prospective largest-input packet uses96GiB available floor and96GiB
RSS/footprint abort, with3GiB runtime available floor. These are selected
operational guards, not source payload caps. Original coding173831 failed;
original P253952 fixture was correctly HTTP400-rejected by64-row admission slack.
Neither becomes a pass under correctedP253888. General capacity/quality/peer
qualification remains in the matrix. Load staging is a separate phase maximum;
retired source allocations are covered only by the declared G assumption.

## Gemma4: numeric refined operating packet

Hybrid/TQ8/codebook8, LCP8GiB, rep1.05, automatic prefill, four full262144
contexts,8192outputs, graft0, verified text-only isolated canonical model locator
with actual projectorfalse. Global cache stride262145 includes its guard row.
The original loaded nibble cache and HB+Hybrid scaffolds coexist; they are not
alternative modes. Feature ON/OFF must retain their actual anchor semantics.

| Named domain | Bytes | Alias/coexistence rule |
|---|---:|---|
| Weights/scalar/load upper |16823270352|Mapped extent upper16817579488 plus independent scalar/staging roles|
| Original cache plus four-slot HB+Hybrid scaffolds |29591901120|Original1438187520 +HB11331216000 +Hybrid16822497600|
| Broad common/non-global queued scratch |12674965216|All sliding seven-ragged-chunk and optional generic overbounds retained|
| Global dense K/V expansion |20803747840|5global layers*4lanes=20pairs; each1040187392B|
| **Named subtotal** |**79893884528**|Not the obsolete35global-pair bound|
| LCP GPU entry/registry envelope |8589934592|Separate from CPU copied anchors; not refunded by CPU quota|
| Server CPU/owned I/O |3221225472|Includes selected1.5GiB process-wide CPU-anchor reservation, not an extra1.5GiB|
| Tools/fixture I/O |3221225472|Original2GiB plus explicit aggregate-target1GiB|
| Client |134217728|Interpreter/control; transcript representations stay in CPU cell|
| G: driver/resource retirement |2147483648|Declared positive assumption|
| Observation age |2147483648|J part1|
| Runtime unavailable headroom |3221225472|Reserve2+reaction1GiB, J part2|
| **Explained operating packet** |**102576680560**|**95.531978235GiB**|

The96GiB floor103079215104 leaves502534544B additional padding beyond ALL
listed positive allowances. It does not assert all filesystem pages reclaimable.
The source decision is in forward_prefill_batched.rs2524–2533/2640–2690/2958:
sliding chunks<=1024 but each global layer consumes a lane's entire remaining
suffix, so global multiplicity is20, not sliding-derived35. P+8192<=262144
bounds each global K/V pair at1040187392B. Per-layer aggregate queries remain
<=4096, not4*4096; common per-query arrays already charge those rows.

Phase dominance retains previous snapshot and handoff joins: two old/new dense
handoffs922746880B each plus two complete dense+Hybrid snapshots5717606400B each
sum13280706560B, below the charged20803747840B global domain. Do not add those
phase-exclusive copies again to the final total. CPU snapshot payloads
(8old/new selected anchors and token ledgers~1273233408B) lie INSIDE ON's1.5GiB
reservation inside CPU3. LCP entries are GPU copies and retain all8GiB separately.
OFF has no invented enforced CPU quota: its finite source/guard assumptions
remain explicit. Load/warmup/cold/single-live/multi-live/decode retain broad
non-global overhead and positive retirement margin; no complete physical
reclamation theorem is inferred. See [four-agent mechanism proof](adr063-gemma-full-input-four-agent-proof.md)
and [matched study profile](adr063-gemma-matched-speed-profile.md).

## Qwen3.8: selected24GiB pool, independently reviewed source candidate

Dense Qwen geometry64layers/h5120/FA24/KV4/D256/V248320, AUTO depth<=3,
TQ8/K8/reset ON, short verifier<=4/all64layers; N4 aggregate prefill<=2048.
The [bounded pool profile](adr063-q5-qwen-pool-growth-profile-2026-10-25.md)
enforces25769803776 cumulative newly created pooled bucket bytes, with no
clear/reset refund. Direct arenas, cache, host payload and projector are separate.
This candidate requires one worker/generation, no GLP/graft/persistence or new
images. OFF/stock cannot borrow the ON ceiling; its pool bound remains separate.

The following retained decomposition is **historical and partial**. Its MTP cell
is replaced below; it is not independently an admission budget.

| Historical selected-source cell | Bytes |
|---|---:|
| Weights including retained CPU values |16799825920|
| Four-slot cache |38332792832|
| Capture old+new transition |5020581888|
| Selected pool ON |25769803776|
| Direct R2048 arenas+embedding+fullprefix bridge |5320737060|
| Old MTP catch-up estimate |436240408|
| Native fallback/descriptor overbound |1412633984|
| Durable full hidden parents |545259520|
| CPU anchors/token/logit envelope |794675200|
| **Historical partial subtotal** |**94432550588 (87.947166141GiB)**|

The new source joins retain conservative phase overcounts:

| Additional owner or correction | Bytes |
|---|---:|
| Auto-loaded projector payload, counted once |927587264|
| Seven additional K8-held FA out_seq roots |352321536|
| Eight distinct K8-held FA linear projection outputs |335544320|
| Tiny verifier out_seq roots, K64/R<=4 |6291456|
| Tiny verifier linear outputs, K64/R<=4 |5242880|
| Replace old MTP estimate with620822552B |184582144|
| Four scalar MTP stages, <=4MiB each |16777216|
| **Closed named source candidate** |**96260897404 (89.649946805GiB)**|

`forward_gpu.rs` retains both helper out_seq and final attention output;
`gpu_full_attn.rs` projects R!=1 into a separate direct[R,h] root. At R2048,
Qwidth6144, eight out_seq roots total402653184B; one current root is already
in the bridge, leaving352321536B. Eight R*h5120 F32 outputs add335544320B.
K-boundary checked drains clear these holds; terminal head wait precedes final
hold destruction. Tiny verification uses R<=4, not K64 multiplied by R2048.
Delayed driver retirement remains in positive G, without completion-time refunds.

`mtp.rs` long catch-up is KV-only, not a full FFN/attention/head transaction.
At R4096/h5120/KV1024, its direct allocation expression is
`7*R*h*4 + 2*R*KV*4 + 16*R + 24 = 620822552` bytes. Shared embedding aliases
base weights; pooled K normalization/RoPE and scalar FFN stay within24GiB.
Scalar direct roots are conservatively padded to four4MiB stages; durable hidden
parents, native workspace and full-prefix bridge already have separate cells.
No4096*vocab output is inferred from this KV-only path.

The header-only projector catalogue reports334tensors(112F16/222F32),
927587264B payload and927606944B file size. The reader stopped at19675 before
payload. `LoadedMmprojWeights::load` preserves F16 and loads the other selected
values as F32; fused QKV views alias roots. Image-free requests do not execute
vision forward, but the auto-loaded projector is not zero. Construction staging
is bounded separately by3*largest tensor47185920=141557760B, joined by phase MAX
rather than added to the full-cache/scratch peak. **The retained F1 identity has
no mmproj identity.** Header metadata is not payload attestation: a fresh projector
hash/stat and conversion-receipt comparison remain pending after the sole runtime
unloads.

| Positive operating cell | GiB |
|---|---:|
| Server CPU/owned I/O |3|
| Tools/fixture I/O, including aggregate fresh targets |3|
| Client overhead, excluding already charged transcript data |0.125|
| Driver/rounding/retirement G |2|
| Observation-age allowance |2|
| Runtime headroom, including reaction/overshoot |3|
| **Operating allowances** |**13.125 (14092861440B)**|
| **Candidate total** |**102.774946805 (110353758844B)**|
| **Prospective rounded available-memory floor** |**104 (111669149696B)**|

The rounded floor leaves1315390852B above the candidate. These are positive
operating allowances, not measured universal OS maxima or a physical-page sum.
Observed competition, pressure, footprint and reaction guards remain applicable.
The104GiB floor is prospective. Independent source review accepted the scoped owner and arithmetic delta; actual qualification remains pending, together
with projector identity; it does not validate earlier90/96GiB launches or admit
a new run. The N1 proposal cannot substitute for this N4 packet. Actual stress,
cleanup and comparator acceptance remain open; the pool experiment's measured
<5%win remains a [negative acceptance result](adr063-qwen38-pool-ablation-negative-result.md).

Private source derivation: `adr063-q5-qwen38-closed-owner-operating-addendum.md`,
SHA256 `34dcafd0ec61a6c61d3007d81318d28ab28a458394e416153f73e406be29f93c`;
projector catalogue basename `qwen38-projector-tensor-directory-v1.json`.
Frozen historical records are unchanged.

Stock/OFF does not borrow the opt-in quota. A separate dense-family retained-bin
role enumeration bounds its selected pool at 13,706,993,664 bytes, rounded to
13 GiB, under the same source catalogue, worker, reset and finite campaign
assumptions. Independent root review accepted that scoped derivation. Because
13 GiB is below the ON packet's 24 GiB cell and the other source owners are
unchanged, the same conservative 104 GiB operating floor can cover either arm
after final identity/settings review. This is source domination, not a measured
stock high-water or physical-release claim.

## DeepSeek4: current bootstrap topology and finite remaining joins

43layers, ratio schedule2uncompressed/21ratio4/20ratio128, hidden4096/HC4,
KVhead1/D512/indexD128, context262144; default prefill rows<=2048/group4,
B4decode2layers/CB. No graft/persistence/calibration/alternate profiling routes.
Bootstrap reuse is eligible pristine startup, not removal of every fifth cache.
See [bootstrap record](adr063-q5-deepseek-bootstrap-reuse-2026-10-25.md).

| Current source-owned capacity / candidate bound | Bytes | Meaning |
|---|---:|---|
| Unique mapped file-page component plus independent F32 |107433539932|107426021376 +7518556; component, NOT complete segmented backing upper|
| Conditional coarse requested-weight upper |107482377724|file107431343264 +1328*2*16384 +7518556; requires actual mapping planner/maxBufferLength assumptions|
| Four full session caches plus128 placeholder |7304294400|4*A262144+A128; not old8205189120 five-large-cache topology|
| Eight compact GPU anchors |142737408|Committed+pending four slots; copy payload, no append-only prefix duplication|
| One old131072 growth source |919617536|Additional to final target cache, one serialized worker migration|
| PREFILL data role-history overbound |10815930368|Conservative pre-query-tile geometric envelope; do not also add endpoint inventory|
| Cooperative sequence-state history |268435456|Separate ordinary pool role history|
| DECODE B4 role-history envelope |916832768|Full43layer/fourlane cycle; not one layer times22CBs|

Cache formula `A(C)=6880*C+17842176`, giving A12818722816,
A131072919617536,A2621441821392896. Swapping loaded/session surfaces aliases
ownership roles, not additional cache roots. Eligible bootstrap saves900894720B.
Ineligible fallback must use old topology. Scheduler8GiB is a shared accounting
limit, not another8GiB allocation. Persistence-off proves image category zero
only for that selected route; if enabled exact imageD1822444296 and3D5467332888
plus index/dirty-I/O staging reopens the packet.

The old2048/group4 endpoint4743790592B is a LOWER inventory, not an all-shape
upper and not another cell above. Query tiles already saved939524096 rounded
bytes there; do not subtract that again from a newer source estimate.

**Remaining concrete source cells to finish before a numeric operating total:**

1. Confirm whole-tensor native mapping plan/maxBufferLength and bounded segment
   count using retained directory metadata. The coarse weight row above is
   conditional; treating107433539932 as a complete upper is unsound. No full
   model payload reread or another model load is necessary for this check.
2. Finish SUBMISSION_INPUT_POOL all-shape bucket history at selected group4.
   Endpoint88183616 is not a supremum; group43's872999568 is another profile.
   CPU-filled inputs remain distinct through whole captured group. ordinary
   GPU-only reset cannot recycle them early. Source seams: verifier_forward.rs
   grouped capture and forward_support.rs alloc_host_input/begin/end scope.
3. Join direct prefill state/embedding/combined output/IdMm with head returns,
   old cache growth and compact anchors using stage MAX. Known direct roots:
   reusable states268435456, embedding167780352, combined134217728,
   IdMm4196352; B4FFNoutputs11272192/combined262144/embedding327696/
   early sparse262144; four-row head2461840. Source-held decode commands<=22,
   output head1; commands reference these same roots, not22payload copies.
   Determine head return multiplicity through sampling/serve and exact CPU
   index Vec capacities (logical max35651584), rather than label them opaque G.
4. Account request-complete clear/refill across the finite28request campaign
   and delayed external owners. Selected DS does NOT install the Qwen growth
   facade. Checked drain plus destruction narrows source ownership but is not
   physical release credit; declare positive bounded retirement/driver G and
   stop/overshoot J with existing live guards. Source-owned graph dispatch/
   binding metadata has finite call cardinality, not an unbounded GPU payload
   copy. No universal per-Arc OS simulator is required.
5. Freeze server CPU/owned-I/O/tools/client allowances and source-stage selection
   for this DS packet, rather than transplant Qwen's known arrays or an old
   12GiB policy. Actual DS diagnostic96GiB preload/124GiB RSS-footprint abort/
   3GiB runtime floor describes operational supervision, not a complete B.

No conservative upper exceeding available memory proves hardware impossibility;
no successful125k growth/N4coding run supplies the missing largest-endpoint
upper. Existing [grammar/N4 evidence](adr063-deepseek-compact-dsml-grammar-2026-10-26.md)
remains scoped to quality/prefix/cleanup. Native kernels and resource factories
belong in mlx-native; selected transaction/worker/pool disposition and source
ledger composition belong in HF. Both published dependencies and exact clean
source/binary identities must be frozen for any changed execution packet.

## Draft disposition and next finite actions

Q36 stock61.451GiB and selected-pool75.451GiB, and Gemma95.531978GiB, are
reviewable explained **operating** totals with positive declared overhead/reaction
allowances. They are not synthetic admission results or physical-fit proof.
Q38 and DS do not yet have a complete numeric disjoint selected-profile packet
in the located source/evidence. Their exact remaining cells are above; runtime
available-RAM policies cannot be substituted for them. Finish those cells,
reconcile each exact source/profile and retained failure, then apply the matrix's
actual stress/recovery/quality gates. No model, build, test, probe or tracker
operation was performed to prepare this document.

## Retained source-derivation provenance

These small private source-review records supply the derivations above. Names
and content digests identify evidence without committing machine paths or data.
Their historical policy/proposal paragraphs are not current activation authority.
This draft corrects arithmetic in its own table only; frozen originals remain
unchanged. The checked-in reports linked above carry public execution scope.

| Retained record basename | SHA256 |
|---|---|
| `adr063-q5-qwen36-numeric-candidate-packet-review.md` | `36f1cc4abb07996d06857713cd1e3573336d8ee7d7a85fcff71ffd832452434b` |
| `adr063-qwen36-stock-pool-finite-role-bound.md` | `23a9ada74f6c133bada31b6e8745434aa9836326fd396bc76a3e1ef1cd6d5b75` |
| `adr063-gemma4-n4-current-source-peak-refinement-v3.md` | `98a147e5a1c417079d4991a464d9944eddd0d5c61148323dc612b72e4e96b9b9` |
| `adr063-gemma4-n4-tools-budget-addendum-v3.md` | `d3d32f0a43df957454826c591b20db5a3bbc9e1ff7836f92b958e727db8e9af6` |
| `adr063-q5-qwen-selected-peak-upper.md` | `a894ac9909ce7ae0d6fad7075a8f84f8a375e7bb355def5f8d3f0284de1a92f4` |
| `adr063-q5-complete-ledger-consolidation.md` | `6508acf276057f16c2c5cfb43410e5963e3b0e38718dcddf4cbeb6878b306475` |
| `adr063-q5-deepseek-selected-scratch-cap.md` | `9299f9c14ff7467bc0b43d46ae85b4815c2f336e5d8e851cdc66943ade35af15` |
| `adr063-q5-deepseek-live-validation-capacity-audit.md` | `5d73726cf273f9c2615db8f9550f94705051d7daf04fcc3f635b01de2463aacc` |
| `adr063-q5-ds-finite-stage-gap-audit.md` | `b9e4d7fadce0c677cda7d2bcf1be5a7f2991f49cfe87b6ad6f1a7bc87f8be945` |
| `adr063-q5-deepseek-pool-reuse-feasibility.md` | `66b0811b5e5d6077f06b20272b0ce6b5bdc18181b6449f1042e5f3ccde66a4dd` |
