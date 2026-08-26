# ADR-049: Agentic state reuse (multi-anchor) and Mixed-phase cooperative prefill

- Status: Accepted; execution in progress (qwen35-family, Gemma4, and
  DeepSeek4 Lane A proof plus the DeepSeek4 B.1 candidate are documented
  through rev 64; Qwen3.6, Qwen3.8, Gemma4, and DeepSeek4 lineage/failure
  gates are green and Lane A family coverage is complete,
  selected-boundary acceleration and scheduler coalescing were falsified, and
  the exact block-segmented replacement has passed its deciding spike, and
  the Qwen rectangular operator-lifecycle correction has exact native-Q5
  state proof, a completed Q5 fused-route falsifier, and a codec-wide exact
  Q4x4 implementation with Qwen3.8 quality/lifecycle and Qwen3.6 route proof;
  its published dependency is pinned, the exact five-format scalar/storage
  matrix is green, and universal four-family two-cycle swap is hardware-sealed;
  the current exact-head physical matrix is reopened after its Q4_K_M
  16-slot cell exposed an optional-checkpoint admission defect, while
  matched-reference acceptance remains open;
  the DeepSeek B.1 and Gemma/Qwen B.2 fail-closed hardware authorities are
  now checked in and model-free green; Gemma and Qwen B.2 implementation and
  coherence are accepted but all affected final performance receipts are
  reopened under the v2 contention authority, while the DeepSeek and Qwen
  current-head real-artifact cells remain open)
- Date: 2026-08-22
- Updated: 2026-08-26 (rev 97 closes the family-wide checkpoint-admission
  audit opened by rev 96 without admitting a new hardware or speed claim.
  Gemma4 and DeepSeek4 now compute the exact prospective family payload before
  any snapshot allocation or copy; zero capacity suppresses only optional
  capture. DeepSeek's short position-zero prompt therefore retains its one
  authoritative full-prompt verifier append when capture is unavailable,
  instead of drifting into two target appends. Its park/reactivate lifecycle
  also reuses the worker-lifetime anchor grant frozen at initial load rather
  than rerunning the available-memory heuristic. Gemma's aggregate ledger now
  includes committed/pending stores, scalar-local rollback, installed bounded
  prefill candidates, and candidates temporarily extracted into a multi-lane
  transaction. Replacement releases exactly the current lane's prior
  candidate before preflighting the new boundary, and the multi-lane ledger is
  recomputed after every install. Stable B2/B4 virtual admission starts from
  that complete live total. Focused source proof is green: 22 DeepSeek serving
  tests, three cross-family capacity tests, and five Gemma anchor tests. The
  complete locked binary suite passes 5,148 tests with 67 explicit real-model/
  benchmark ignores, and `cargo check --locked --all-targets --all-features`
  is green; exact-artifact physical and matched-current-peer timing receipts
  remain open. Rev 96 records and
  acts on the exact-head physical
  multi-slot falsifier at hf2q `08ef2e35`. BF16 passed scalar-equivalent
  1/2/4/8/16-slot execution and Q4_K_M passed 1/2/4/8, but Q4_K_M width 16
  returned HTTP 500 after a 2,435,398,041-byte immutable grant made
  `K_effective=0`: the rectangular planner proved that four transient
  checkpoint reservations fit the currently empty stores, then the staging
  path applied the stricter all-configured-slot depth rule and rejected every
  lane with `NoCommittedCapacity`. The synthetic checkpoint-suppressed
  rectangular test failed at the executor's plan-identity guard, proving that
  optional state reuse had been coupled to otherwise-valid target execution.
  The corrected contract keeps checkpoint admission fail-closed without making
  generation fail: capacity-aware planning suppresses capture before GPU work,
  the same rectangular target transaction runs without a checkpoint, and an
  unexpected post-submit staging rejection discards every cohort-local pending
  checkpoint before committing the already-validated target advances when the
  rejection is capacity-only. `PendingOccupied` is not capacity pressure: the
  single-worker planner proved every selected store had no pending payload, so
  that outcome increments an invariant-failure counter and rolls the cohort
  back fail-closed. Planned capacity skips become capture telemetry only after
  the rectangular target advances commit, preventing cancelled or rolled-back
  work from creating phantom captures; the partial-capacity counter retains
  its documented depth-or-simultaneous-pending predicate. It does
  not reject the requested slot count, inflate the worker's logical grant, or
  overpromise unreserved host memory across park/reactivate and model swap.
  Exact-head physical and matched timing receipts remain open until this change
  lands and the full chain is rerun. Rev 95 closes an inherited-process-group proof defect
  across the remaining Qwen rectangular/Mixed, Gemma B.2, and DeepSeek B.1
  performance leaves before another model run. Each leaf now self-reexecs
  through the release-gate supervisor, proves that its stable owner PID is its
  process-group leader before model admission, and records that same owner in
  every settle and measurement sample. Qwen and DeepSeek exempt only their
  exact verified server PID; Gemma retains its fixed-grid multi-process
  identity checks under the isolated owner group. The shared thermal helpers
  now forward that optional exact-server identity. Every final receipt binds
  process-group-cpu-v2, the 100% foreign-CPU ceiling, owner scope and PGID,
  and continuous sampling; independent verification joins every raw
  six-column contention row back to the sealed owner. Forced-sentinel tests
  reject inherited groups before reading model arguments, and resealed policy,
  threshold, owner, continuity, and middle-row-owner mutations all fail.
  Qwen's model-free contracts, Gemma's 37-mutation battery, DeepSeek's
  23-case battery, the shared thermal contract, and ShellCheck are green.
  This revision changes proof authority, not model math, and admits no new
  hardware timing. Rev 94 re-anchors the still-open matched comparison to
  the operator's current clean peer source at `5e6a37cb`, one commit beyond
  the prior pin. The intervening upstream diff is Vulkan-only and therefore
  does not change the Apple execution path, but exact source and runtime
  closure remain part of performance authority. Build 10639 retains the same
  launcher SHA-256, changes the complete Mach-O closure hash, and regenerates
  all 24 quantizer fixtures byte-identically. No model measurement from the
  older pin is accepted for current-head closure. Rev 93 closes three
  shell-semantic falsifiers before
  spending the reopened current-head hardware run. The physical preflight's
  negated listener pipeline could fail and then return success after later
  quiet-host checks; both matched runners' calibration functions could also
  mask a failed power, thermal, or contention sample when called beneath a
  conditional because Bash suppresses `errexit` throughout that function
  body. One shared port predicate and one shared calibration writer now
  propagate every failure explicitly, with injected occupied-port, power,
  thermal, contention, and output-write failures. Finally,
  process-group-cpu-v2 again preserves all of the named prohibitions declared
  by its launch receipt: compiler and native inference processes are rejected
  unconditionally, full command capture distinguishes Python model work from
  unrelated Python utilities, and aggregate foreign CPU remains independently
  capped. Behavioral mutations cover every named runtime and both Python
  outcomes. Current-head hardware acceptance remains open. Rev 92 closes
  three integration falsifiers found while
  reviewing rev 91. A measured leaf no longer trusts an inherited process
  group: it re-execs through the release-gate supervisor, proves its stable
  owner PID is its PGID before reading model arguments, and uses that owner for
  every sample while exempting only the exact live server PID. The sealed
  summary binds that owner PGID and reopened validation joins every preflight,
  settle, and measurement row to it. Both matrix validators now require the
  exact process-group-cpu-v2 child contract; a relabeled 16-log legacy receipt,
  weakened threshold, missing owner, or non-continuous policy cannot pass.
  Finally, reopening is semantic rather than hash-only: all expected trial
  paths and 24 or 120 calibration logs per child are revalidated for cadence,
  state, alignment, owner identity, exact manifest membership, and non-symlink
  evidence roots after their hashes pass. A mutation
  changes a middle measurement row to `contended`, recomputes both seals,
  proves the hash-only seal remains valid, and proves semantic reopening
  rejects it. Current-head hardware acceptance remains open. Rev 91 closes
  the same contention-proof gap in both
  Qwen3.8 matched-reference authorities before spending the current-head
  hardware run. Their 120-second nominal settle and continuous measurement
  monitor still called the obsolete name-only heavy-work predicate, so a
  generic browser or non-model CPU workload could again be mislabeled quiet.
  Both the single-user and five-width physical ABBA runners now record
  process-group-cpu-v2 samples on the exact thermal timestamps, require quiet
  at measurement start, continuously, and at measurement end, validate the
  settle/measurement logs and thermal alignment, and bind the 100% aggregate
  foreign-CPU ceiling into their sealed summaries. The old name-only parser is
  removed. Source mutations prove that stale policy, a weakened threshold, or
  non-continuous sampling cannot seal; current-head hardware acceptance remains
  open. Rev 90 closes a shared performance-proof falsifier
  before the next model load: the process-group-v1 contention guard reported
  `quiet` while an orphaned browser tree and unrelated analysis consumed
  about 7.7 foreign CPU cores. The accepted process-group-cpu-v2 receipt
  excludes the complete owned gate group, preserves unconditional compiler/
  peer rejection, records aggregate foreign CPU, and treats one full foreign
  core as contention. Model-free boundary, aggregation, schema, and stale-v1
  mutations pass. Because the observed orphan predates the rev-86 Gemma and
  affected August 25 Qwen measurements, those implementation/coherence
  decisions remain accepted but their final performance authorities must
  rerun under v2. Rev 89 closes the
  pre-run proof audit for the reformed
  DeepSeek B.1 workload: every decoder now proves an exact warm-cache plan and
  matching client-visible cached-token usage; every long prefill proves a cold
  plan and zero client-visible reuse; all four runtime prefill admissions must
  precede the first decoder completion; and rebound canaries exercise those
  relations plus the bounded-Mixed/unbounded-pure discriminator. Rev 88
  records the second DeepSeek B.1 hardware
  falsifier: the corrected warmup proved that the nominal live decoders were
  actually a cold cohort, so the cold-wave admission barrier correctly held
  the following prefills until decode completion; the observed 4x512 event
  was pure prefill, not an ignored OFF policy. No sample was admitted. The
  rerun primes stable decoder prompts, requires nonzero cache reuse, and binds
  cooperative receipts to bounded-Mixed versus unbounded-pure execution.
  Rev 87 recorded the first DeepSeek B.1 hardware
  falsifier: the warmup reached the real eight-slot runtime, then the producer's
  request-id parser failed to compile because a shell-continuation backslash
  was embedded in its Perl program. No performance sample was admitted; the
  parser is corrected and executed by the model-free contract before rerun.
  Rev 86 recorded the accepted exact-lineage Gemma product
  rerun: nominal 128/192 retain statistically supported product gains, while
  nominal 256 proves the zero-rectangle scalar fallback exact and non-inferior.
  Rev 85 recorded the completed eight-pair Gemma product
  falsifier and the resulting serving policy: nominal 128/192 cells passed the
  immutable 1.05 lower-confidence speed gate, while nominal 256 did not. The
  serving planner therefore caps stable rectangular boundary work at 192 rows
  and retains the equality-proven scalar path above it; the next product rerun
  must prove both the profitable rectangles and the 256-row fallback. Rev 84 recorded the second fail-closed Gemma product
  calibration: one payload word rendered 105 uncached rows; combined with rev
  83's 24-word/151-row receipt, the exact relation is 103 fixed tool-envelope
  tokens plus two tokens per payload word. Nominal 128/192/256 cells therefore
  render 127/191/255 rows and remain inside both the ±4 gate and the production
  256-row ceiling. No performance sample has yet been accepted. Rev 83
  recorded the clean-source Gemma stable-route
  authority across both native KV representations and the first realistic
  product-gate falsifier: packed-V and full-F16 each passed all 29 exact
  B2/B4, tail, deep-cache, wrap, selected-state, unselected-slot, and
  continuation cells, while the product producer rejected its first nominal
  64-row sample because the hardened assistant-tool-call plus matching-result
  envelope rendered 151 uncached rows. That first point disproved the stale
  calibration and required the second point recorded by rev 84; the real
  tool-turn cells must rerun before any product speed conclusion. Rev 82
  recorded that the exact shared-weight broadcast matmul
  hypothesis is falsified and fully removed from hf2q: the published backend
  primitive preserved byte-exact B2 state and continuation but reduced the
  interleaved three-sample B2 median only 0.752%, from 109.708 ms to
  108.883 ms, missing the predeclared 5% gate. The dependency pin, production
  branch, and spike-only tests were removed; the canonical rectangular route
  remains the accepted implementation. Rev 81 recorded that the direct
  production-route attention performance hypothesis is falsified and fully
  removed: its first B2 global
  D512 packed-TQ equality case differed from the current tiled-F16 route, so
  no timing was taken and no production attention code survives. Rev 80
  recorded that the checked-in Gemma B2/B4 M32 authority
  proves exact final logits/tokens, direct byte equality for every selected
  slot's complete logical hybrid-cache state and cursor/layout metadata, full
  seeded unselected-slot preservation, and exact one-token continuation. The
  B2 rectangle reduced measured suffix wall from 126.401 ms to 111.944 ms
  (11.4%); B4 reduced 245.089 ms to 194.658 ms (20.6%). Product-serving and
  matched-reference gates remain open. Rev 79 recorded that the minimal Gemma output-projection plus
  expert-gate/up lane correction is exact through final logits for the pinned
  B2 M32 Q5_K_M hardware oracle and improves the 64-token rectangle from
  125.6 ms of sequential scalar work to 103.8 ms, a 17.4% wall reduction and
  about 21.0% aggregate-throughput gain. The corrected first divergence also
  restores exact downstream expert-down/combine output without changing those
  operators; product-serving remained open.
  Rev 78 recorded that the Gemma live-rectangle stage bisection
  proves the existing production path exact through attention and proves the
  canonical per-lane output-projection candidate exact through post-attention
  state, all raw dense/router
  projections, GELU, every top-k expert ID and routing-weight bit, and dense
  down; the first changed operator is the aggregate expert gate/up `mm_id`
  dispatch. The implementation hypothesis is therefore lane-width expert
  gate/up first, with an explicit shared-scratch barrier, while expert down is
  decided only after that correction. A proposed attention replacement has no
  demonstrated correctness defect and its first TQ oracle did not reproduce
  the live tiled route; it is reduced to a current-route exactness/performance
  falsifier before any production code may survive. Rev 77 sealed the exact
  five-format physical matrix and universal
  four-family swap green; universal SlotAware private-queue liveness is
  hardware-proven for Gemma's four-slot burst and the Gemma B.2 runner now
  rejects the full single-quoted continuation bug class;
  cross-family swap harness identity race and post-drain rehash falsified;
  canonical schema-v2 preflight, isolated-Cargo, and source-dominant
  host-wired authorities are model-free green; Gemma's canonical Stable-chat
  aggregation route is now correctly classified as OPEN after two hardware
  falsifiers rejected both tiny-cue batching and concat-shaped boundary
  batching on coherence; a synchronized layer scan and exact binary row dump
  now locate the first concat-shape divergence at layer zero's quantized output
  projection; exact-path mapped-artifact
  classification is model-free green after falsifying a basename collision)
  — the Qwen implementation lineage begins at `95d618c8`, based on main
  `32181b61`: explicit per-slot AnchorStore,
  linear-lineage pruning, fail-atomic restore preflight, exact payload
  ownership/accounting, terminal publication, A.8 logs/metrics, idle audit,
  independent reference-model + 17-mutation battery, and Lane C corrections.
  Qwen3.8 now has BF16 and Q4_K_M width-four SlotAware lineage/cancellation/
  failure-injection receipts. Rev 20 corrects the Q4 gate's transport fixture:
  the former 2.7 MiB body was rejected by Axum before model admission, while
  the token-dense replacement reaches hf2q and returns the exact HTTP 400
  `context_length_exceeded` contract. The selected-boundary performance sweep remains
  useful falsification evidence, but source-route review invalidated the
  candidate's universal equality claim and its runtime integration was
  removed. Rev 21 adds the required Qwen3.6 shape. Rev 22 adds the distinct
  Gemma4 hardware proof. Rev 23 adds the distinct DeepSeek4 hardware proof and
  closes Lane A family coverage. Codec/physical-width coherence is a separate native-format
  gate; the semantic anchor gate is not multiplied into a codec-by-width
  Cartesian product. See the execution ledger below. Gemma4 parity in this revision is based on main
  `2c6bcb61`: the model-neutral state machine is extracted into
  `src/serve/api/anchor_store.rs`; Gemma retains its family-native sliding-KV
  payload, image fingerprint matching, independent byte policy, and telemetry.
  Its model-free gates prove fail-atomic all-layer restore preflight, terminal
  pending publication, cancellation, lineage rewind, exact owned bytes, and
  native equality hits. The rev-11 audit also closed a single-request Gemma
  failure path that reset physical KV after rollback-checkpoint capture failed
  but retained committed anchor authority; reset now clears retained tokens,
  committed anchors, and pending capture together, with a model-free
  regression. The real-artifact Gemma gate is now green at `c0be27fd`; its
  exact receipt is recorded in §A.6. DeepSeek4
  parity in rev 8 is based on the generic AnchorStore checkpoint `234fb394`:
  recovery-tail snapshots are capacity-portable across cache growth, every
  committed and pending payload is preflighted before the live cache migrates
  once, and strict-prefix-only anchor matching preserves the family-native
  live-logit equality path. Its real-artifact receipt is now green and recorded
  in §A.6.
  The model-free B.0 Mixed-step spike also passed both cooperative-prefill
  commit and poison paths without changing any concurrently decoding peer
  cursor or cache/compressor byte; the exact ownership ledger and test names
  are recorded in §B.0. B.1 now attempts a FIFO-prefix cooperative prefill
  capped at 128 rows per lane while retaining the two-window serial fallback
  and recovery-tail priority. Its model-free cap/workload gates and
  production-event latency receipts are recorded in §B.1; hardware latency,
  artifact parity/performance, thermal, and memory acceptance remain open.
  Rev 18 closes the A.8 model-free observability gap with one fixed-schema,
  per-attempt outcome event shared by Qwen, Gemma, and DeepSeek. Publication
  disposition is stamped only when a pending anchor is actually published;
  transient rollback anchors remain explicitly unpublished. Qwen and Gemma
  cancellation now prune restored descendants before reuse, all Gemma stable-
  batch lanes are reset and classified independently after cohort failure,
  and restore errors are distinguished from cleanup errors. The event reports
  a slot identifier where the engine owns one, matched-old-tail divergence,
  actual per-slot peak committed+pending bytes, final pending discard, capture
  time, and prune disposition. Rev 18's focused model-free tests are green; it
  made no new real-artifact or speed claim.
  Rev 19 corrects four source-audit defects in that contract: lifetime peak
  bytes now live on each `AnchorStore` instead of being read from a family-wide
  maximum; DeepSeek SlotAware identity follows the slot through its shared
  execution swap; DeepSeek cancellation retains the pre-restore pending-discard
  fact and emits one terminal event on restore and reset paths; and a Gemma
  stable-cohort failure retains prune/discard facts from lanes that completed
  pruning before a peer failed. Production-callsite regressions cover event
  cardinality and values. The focused AnchorStore and family restore-failure
  tests are green after integration; whole-tree and remaining hardware gates
  stay tied to the final landing commit.
  Rev 22 also closes a canonical Gemma startup blocker found by the hardware
  gate: the activation allocator created `argmax_params` as two F32 values,
  while the loader wrote one U32 and the native operator correctly rejected
  the mismatched buffer. Production now allocates exactly one U32, a Metal
  canary pins that contract, and synchronous warmup logs retain the complete
  error chain. The corrected server mapped every ordinary matrix from the
  artifact without anonymous matrix storage and completed the full four-slot
  lineage, cancellation, injected-restore-failure, cold-recovery, and rebuilt-
  reuse gate. The gate's cancellation expectation is deliberately slot-local:
  one queued sibling inherits the cancelled owner's checkpoint while siblings
  admitted to other free slots may prefill cold. Cross-slot checkpoint sharing
  is the separate A.7 hypothesis, not a hidden Lane A requirement.
  Rev 23 also closes two canonical DeepSeek artifact-load regressions exposed
  by that gate. The loader had assumed hash-routed expert IDs might repeat and
  therefore rejected Q2_K's grouped expert route before reading the artifact;
  the loaded `tid2eid` payload is now the authority, with a duplicate-safe
  MV/flattened-down fallback and out-of-range rejection. The served artifact's
  387,840 hash rows were all unique and in range. Separately, a universal
  native-storage refactor had accidentally included elementwise F32-consumer
  state in the zero-copy matrix arm, contradicting the loader's own contract
  and rejecting Q8_0 compressor APE tensors. Native matrices remain untouched;
  only quantized elementwise-only state expands once to F32, with no
  requantization. Complete loader error chains are now preserved at the
  hot-swap boundary.
  Rev 24 records the source-coherent worker half of FreeToken-inspired elastic
  parking: drained Qwen/Gemma/DeepSeek workers can release a checked registered
  mutable-runtime set while retaining immutable mapped weights, reject work
  until cold reactivation, and require full eviction when a park reply is
  indeterminate. This is not yet pool credit or a swap-speed claim: family-
  helper proof, separately charged manager reservations, fresh-generation
  publication, and A→B→A hardware comparison remain open. Rev 24 also records
  the first Qwen B.2 hardware falsifier: the gate correctly rejected its own
  uncalibrated 128-row sample and buffered trace log before any performance
  decision. The corrected gate subtracts the measured 36-token rendered
  envelope and proves a live one-transaction trace before thermal settling.
  Rev 25 records the resulting valid seven-trial measurement without promoting
  it: the point estimate supports the opportunity, but its lower fixed-share
  confidence bound misses the pre-registered confirmation threshold by one
  percentage point. The only permitted precision extension is now fixed at 21
  trials per width with identical widths, ordering, settings, and decision
  thresholds.
  Rev 26 records that extension as terminal confirmation. Qwen35-family
  implementation therefore opens at a rectangular, slot-aware target batch:
  compatible equal-row lanes retain one aggregate activation matrix through
  dense/MoE projections (including one `mm_id` dispatch), while attention,
  DeltaNet state, rollback, checkpoint publication, and final logits remain
  explicitly lane-mapped. Layer-level batched-vs-scalar state/output parity is
  the next falsifier and precedes scheduler publication.
  Rev 27 records the first layer spike rather than promoting it. Rectangular
  recurrent DeltaNet executed four 128-row sequences through the native
  `n_tokens × n_seqs` axes and matched four scalar forwards' output, recurrent
  state, and convolution state bit-for-bit. The sparse reordered physical-slot
  fixture left an unselected warm peer unchanged, and the new path rejects the
  unsafe chunk-scan experiment before mutation because that would change the
  scalar route. The first full-attention draft
  was falsified by source-route review before scheduler integration: it widened
  the byte-packed TQ decode-vector dispatcher, while a production 128-row
  scalar prefill uses the BF16 tiled prefill dispatcher over freshly projected
  K/V. That draft was removed. The revised spike preserves the tiled route:
  aggregate projections remain one row matrix, existing per-lane permutations
  stage `[batch, heads, rows, head_dim]`, one native tiled dispatch uses its
  existing batch axis, and banked TQ writes remain explicit. Resumed cohorts
  additionally require a shared cursor, K length, capacity, and scalar route;
  incompatible lanes retain the scalar path rather than changing arithmetic
  or leaking cache state. The replacement tiled-attention bridge passed the
  measured-shape `2 × 128` gate with every F32 output bit identical to two
  scalar tiled forwards.
  Rev 28 closes the full-attention layer falsifier at widths two and four,
  both with 128 distinct rows per lane and sparse reordered physical slots.
  Aggregate RMSNorm/Q/K/V/gate/output projections, one native tiled-attention
  batch, and banked TQ writes produced bit-identical scalar outputs, packed K/V
  bytes, norm bits, and cursors; unselected slots remained unchanged. Duplicate,
  out-of-range, warm, and under-capacity cohorts reject before any dispatch,
  commit, cursor update, or cache-byte mutation. The next gate moves one level
  up: complete dense and MoE target forwards, output-head/MTP state, and the
  all-lane transaction/publication contract.
  Rev 29 closes the complete model-free target falsifier. A dedicated fresh,
  text-only rectangular entry keeps embeddings, residuals, dense/MoE FFNs,
  and native artifact-weight representations on one sequence-major
  `batch × 128` row matrix, while only the proven full-attention and DeltaNet
  helpers map lane state. Widths two and four matched independent scalar dense
  forwards bit-for-bit for final logits, every normalized `h_nextn` row,
  complete cursor-visible cache snapshots, and one subsequent ordinary decode
  token. The width-four quantized-MoE fixture passed the same comparison and
  retained exactly three aggregate ID-routed projection calls per layer—not
  per lane or expert. Sparse reordered slots and a warm unselected fifth slot
  remained exact. Every selected slot is captured before GPU work; any target
  or output-head error rewinds the whole cohort's cursors and DeltaNet parity.
  Admission rejects recovery capture, the single-sequence chunk-scan
  experiment, non-TQ full-attention state, and host-loop F32 MoE before cache
  mutation. This is correctness evidence, not a shipped scheduler or speed
  claim. The next gate is worker admission plus Qwen3.8 lane-local MTP,
  all-lane publication/failure injection, and exact-artifact performance.
  Rev 30 corrects the admitted row shape from one synthetic 128-row point to
  the scalar tiled-prefill range `16..=128`. The deciding production trace has
  a 121-token stable prefix followed by a seven-token tail inside the 128-row
  scheduler allowance; admitting only 128 rows would therefore miss the
  measured agentic workload. Four sparse, reordered 121-row lanes now match
  independent scalar target forwards bit-for-bit for logits, every normalized
  `h_nextn` row, all live-prefix cache bytes, and one-token continuation. Rows
  below 16 remain scalar because the authoritative D256 fresh-attention route
  changes below that threshold. The scheduler must end a cohort at the common
  stable boundary, publish its checkpoint transactionally, and leave the
  residual tail to the ordinary scalar route.
  Rev 31 lands that worker-owned transaction as a model-free implementation
  candidate. The immutable worker-lifetime `HF2Q_CROSS_SLOT_ADMIT` policy now
  admits the largest already-runnable compatible FIFO prefix at widths four,
  three, then two; it never waits for another lane or skips an incompatible
  earlier request. Admission is limited to cold text lanes with the same
  `16..=128` stable boundary, identical target/MTP route, cold target and MTP
  cursors. Checkpoint capture is admitted independently: all-lane K+1
  reservations must fit the aggregate anchor budget before mutation when
  capture is enabled, while insufficient optional checkpoint capacity keeps
  the same rectangular target transaction with capture disabled. One
  rectangular target transaction retains
  aggregate dense/MoE projections; Qwen3.8 AUTO performs the existing exact
  MTP catch-up lane by lane under the same supervisor lease. A catch-up failure
  rewinds every target and MTP slot, marks every retry state speculation-
  unavailable, and replays one ordinary target cohort without hidden capture.
  The worker validates every checkpoint, stages every pending anchor, rechecks
  cancellation and FIFO ownership, and only then commits physical state and
  advances scheduler ledgers. A post-submit capacity rejection discards the
  complete pending set but preserves the already-validated target advances;
  a non-capacity staging result is an invariant failure. Any validation,
  invariant staging failure, cancellation, or injected post-checkpoint failure
  discards the complete pending set and rolls every selected slot back. A
  cancellation terminates only closed lanes and
  reinstalls open peers cold without ledger advance; validation, staging, and
  injected failures fail closed after reset. The canonical Qwen3.6 launcher
  enables the policy by default and the Qwen3.8 launcher inherits it; setting
  the switch to zero before worker startup is the matched serial-control path.

  Focused source gates at this revision are green: 38 bounded-prefill/
  watchdog tests, 17 tests selected by the rectangular filter (one separate
  vision hardware test ignored), seven direct cohort-transaction tests, the
  declared-failpoint wiring test, locked
  `cargo check`, launcher syntax validation, and `git diff --check`. The
  synthetic D256 state executor proves exact two-lane 121-row checkpoints,
  all-lane retry rollback, an untouched unselected peer, capture-disabled
  Qwen3.6 parity, and rollback after the post-checkpoint failpoint. Its native
  D256 MTP fixture also proves that a recoverable catch-up failure rewinds both
  lanes and performs exactly one ordinary replay matching the ordinary target,
  while a nested typed command-buffer failure preserves its cause, rewinds all
  lanes, performs zero replay, and fans out every owned cohort reply once. The
  published event distinguishes `NotRequested`, `Succeeded`, and
  `OrdinaryReplay`, and the FIFO canary prevents skipping an incompatible or
  closed middle lane. These are model-free correctness receipts, not an exact-
  artifact, serving-quality, or performance claim. Direct Qwen3.6/Qwen3.8
  OFF/ON hardware gates remain the next falsifier.
- Owners: hf2q serving engine (execution: the active qwen35/qwen38 serving-lane session; plan authored by the FreeToken research session)
- Code pins: planning review at hf2q `242882e8`; rev-6 execution based on
  merged main `32181b61`; current execution pins mlx-native `0.15.0`. Anchors were authored at
  `815bd48d`; every correction-touched anchor was re-verified before editing.
- Provenance: full paper+code study of FreeToken (arXiv 2608.16157, "FreeToken: Efficient Edge-Native MoE Serving with Bandwidth-Adaptive Execution") mapped onto hf2q/mlx-native by a nine-agent research swarm, then adversarially reviewed by two independent external models (Kimi K3 via opencode; gpt-5.6-sol via codex, 516k-token source-grounded review). Both reviews' MUST-FIX items are incorporated; the gpt-5.6-sol review found and this ADR closes a stale-KV lineage coherence bug in the original draft (§A.2).

## Context

FreeToken serves frontier MoE models on consumer discrete-GPU PCs by treating host RAM as the expert-weight source of truth and VRAM as a global (layer,expert) LRU cache, splitting decode misses between PCIe-fetch and CPU-execute by a measured closed-form ratio, streaming whole expert layers double-buffered during prefill, checkpointing hybrid-model recurrent state at chunk boundaries nearly free, and resizing pools elastically at idle safe points. Its headline agentic result: worst-case TTFT under 44 s where llama.cpp/Ollama/KTransformers show 232–946 s worst turns — earned mostly by recurrent-state checkpoint placement plus prefix reuse, not by raw kernel speed.

Most of FreeToken's bandwidth machinery does not transfer to Apple Silicon unified memory (§Rejections). What does transfer lands exactly on hf2q's two open sore spots:

1. **Same-slot context edits go cold.** Agent harnesses (opencode, Claude Code) rewrite context mid-conversation — strip thinking blocks, collapse tool output. hf2q keeps exactly one anchor per slot (`prompt_anchors[slot] = Some(...)`, install at src/serve/api/engine.rs:16925), so any divergence inside retained tokens that predates the single anchor recomputes the whole prefix. hf2q's *matching* is already ahead of FreeToken's (template-aware cue-less re-render, strict-token-prefix acceptance, src/serve/api/handlers.rs:1905-1938, vision-aware via `expand_stable_prompt_boundary`); its checkpoint *depth* is 1.
2. **DeepSeek4 Mixed-phase prefill did not aggregate rows before B.1.** The
   prior path bypassed the landed cooperative cohort whenever decode was
   runnable, so each serial 256-token slice re-paid the measured fixed cost.
   B.1 now runs decode first and then attempts a FIFO-prefix cooperative cohort
   capped at 128 rows per lane; its real-artifact speed, parity, SSE-tail,
   thermal, and memory contracts remain open.

Key enabling facts at HEAD:
- The per-slot anchor payload `HybridKvSlotAnchor` (src/inference/models/qwen35/kv_cache.rs:996-1002) is cursors + DeltaNet state only — **zero KV bytes copied** (full-attn K/V is append-only; the slot's own KV is the pin). Snapshot and restore are **already slot-indexed** (`snapshot_slot_anchor` kv_cache.rs:1532, `restore_slot_anchor` :1607) with the cursor proof `live_cursor >= saved_cursor` (:1651-1656) and per-slot ping-pong parity re-canonicalization (:1694).
- One anchor costs ≈ 62.8 MiB for Qwen3.6-35B-A3B (30 DeltaNet layers × (2 MiB recurrent [128×128×32 f32] + 96 KiB conv [8192×3 f32])); ≈ 149.6 MiB for Qwen3.8-27B (48 layers). Capture is a ~2–4 ms host memcpy at a point where the engine is already stopped on the boundary. Anchors are host-owned `Vec<u8>` — anonymous RAM, not Metal working set.
- The prefill chunker already clamps a transaction to end exactly at the stable boundary (`qwen35_next_prefill_end`, src/serve/api/engine_qwen35.rs:3635-3648; checkpoint emission gated on `stable_prompt_prefix_tokens == Some(end)`, :3966-3985). Since commit `35e42b28`, single-slot workers use a 4,096-token transaction ceiling while multi-slot stays 2,048 (`qwen35_slot_prefill_chunk_tokens`, engine.rs:16729-16749).
- DeltaNet spec-decode rollback (`rollback_la_to`) is per-slot in signature (kv_cache.rs:2296-2300) but structurally absent from the slot-aware worker call graph (pin H38, engine.rs:44835-44906) — prefill-time anchors face no speculative interleaving today.

## Decision

Three lanes, in this order. Lane A is the primary value; Lane B is gated on a coherence spike; Lane C ships with Lane A's first PR.

**Scope directive (Robert, 2026-08-22): every lever this ADR ships is a cross-family benefit — all supported models, all supported families share it. Single-model or single-family shipments are milestones, never the deliverable, and the ADR is not complete while any supported family lacks a shipped lever it can benefit from.** For Lane A: the first implementation lands on the qwen35-family engine (one engine serving both Qwen3.6-35B-A3B and Qwen3.8-27B); gemma4 and deepseek4 parity are REQUIRED phases, not optional follow-ons; per-family gates run on the artifact each lane actually serves. For Lane B: deepseek4 is the first implementation and §B.2 carries the required family-generalization evaluation. The directive also binds future families: any family gaining serve support later must adopt these levers as part of its engine bring-up, not as a deferred extra.

**Not in committed scope, but an OPEN HYPOTHESIS with a deciding spike** (framing per Robert 2026-08-22: these are questions needing data, not parked scope): raising `MAX_COOPERATIVE_PREFILL_ROWS` (the "Lane 2b" of the draft). Hypothesis: a larger aggregate row budget still pays in pure-prefill waves without breaching the memory envelope. Data against so far: ADR-042 records a 4,096-row OOM (ADR-042:59, :503) and a later 4,096/cold-cooperative failure (:2069); projected gain ~1.15× confined to the pure-prefill wave; receipt verifiers/artifact tests encode the exact 2,048-row shapes. Deciding spike: Lane B's wave-phase profiling (does F-dominance survive in pure-prefill waves once Mixed cohorts land?) plus a fresh transient high-water measurement at 4,096 beside the 100 GiB artifact under the current single-layer-CB cooperative structure. Outcome: implement, or falsify and record.

### Lane A — Multi-anchor slot-local checkpoints (qwen35 first)

**Contract (per gpt-5.6-sol M3, smaller than the draft's):** one validated current-turn boundary per *successfully committed* request, accumulated over observed turns into a per-slot anchor store. No sorted-boundary-set machinery, no seeding of historical turns from a cold transcript (that would require handler/template work to render and validate multiple message prefixes — explicitly out of scope). The handler already computes exactly one boundary per request (handlers.rs:1905-1938 → `SamplingParams` → `Qwen35PrefillState`); the engine change is to *accumulate* instead of overwrite. Default-on: the workload is 100% agentic, host capture is a few milliseconds, selected-row capture removes the extra internal stop when route equivalence is proven, and the immutable aggregate grant derives actual depth from the artifact-and-slot costs in A.4 instead of assuming one universal byte figure.

**A.1 — AnchorStore.** Replace the single `Option<Qwen35PromptAnchor>` (struct at engine.rs:16865-16871) with an explicit per-slot store, not a raw Vec:

```
AnchorStore { committed: Vec<Anchor>, pending: Option<Anchor>, lineage_epoch: u64, owned_bytes: u64 }
```

Three-state publication machine (gpt-5.6-sol M2; generalizes DeepSeek4's pending→committed two-phase commit at engine_deepseek4.rs:921-946):
- *Committed* anchors: visible to affinity and cancellation rollback.
- *Request-local pending* capture: invisible to other admissions until the request reaches terminal cache+ledger success (the retained-token ledger publishes at engine.rs:19148 — pending merges atomically there, then eviction applies).
- On failure or cancellation: discard pending; the committed list survives unchanged. The existing install sites (engine.rs:18825, :18870) sit in the same match arms as cancellation recovery (:18829-18841, :18872-18881) — recovery must additionally prune every committed anchor whose cursor exceeds the post-recovery live cursor (Kimi M2).

**A.2 — Linear-lineage invariant (the coherence core; closes the draft's stale-KV bug).** Anchors index positions in ONE mutable per-slot KV log. Restoring anchor A and then writing a divergent suffix overwrites the physical rows that backed every deeper anchor — after which a deeper anchor's token match AND cursor check can both pass while the KV bytes are wrong. Therefore, fail-closed law:

> Before the first KV write after restoring anchor A, invalidate every anchor deeper than A (bump `lineage_epoch`; drop or tombstone the descendants). A cold reset, a slot poison, or any FAILED restore invalidates the ENTIRE store for that slot — mandatory for fail-closed recovery. Anchor selection must check epoch, never tokens+cursor alone.

Mandatory regression (must exist before any restore path merges): build lineage A→B→C, restore A, prefill divergent branch X, then send a request matching old C — the engine must go cold (or restore A), NEVER restore B or C. Byte-compare the divergent-branch output against a cold run.

**A.3 — Restore-on-divergence.** Extend slot affinity (`qwen35_slot_affinity`, engine.rs:16975-17047) from best-of-{live cursor, one anchor} to best-of-{live cursor, deepest *epoch-valid* committed anchor whose tokens are a prefix of the request}. Matching preserves **equality** (not strict prefix): the existing full-prompt-equality path replays stored `prefill_logits` and skips the forward entirely (engine.rs:16985, :17001) — that behavior generalizes to the deepest anchor equal to the full new prompt. On divergence inside retained tokens: `restore_slot_anchor` at the selected anchor, re-prefill only the suffix. Today's behavior in that case is cold; this is the lane's entire payoff.

**Restore contract (fail-atomicity — executor-audit finding, verified at `242882e8`):** `restore_slot_anchor` currently interleaves validation with mutation — full-attn cursors are rewound inside the same loop as the per-layer ensures, and the MTP cursor is rewound before the linear-state copies, which can still fail — so a mid-restore error leaves the slot partially rewound. Lane A must refactor it to **preflight ALL validations, then mutate**. On any restore error: hard-reset the slot and clear its entire anchor store — NEVER fall back to a shallower anchor after a partial restore (that is the A.2 bug class by another road).

**A.4 — Eviction & budget.**
- Eviction: positional keep-newest-K (K default 4; anchors form a nested prefix chain, so LRU-by-restore is actively wrong — a twice-edited turn would evict the deeper anchor about to be needed; both reviewers concurred). Descendant invalidation (A.2) runs before any eviction policy matters. One refinement permitted later, telemetry-first: reserve slot 0 of the list for the oldest/system boundary, K−1 for newest.
- **Payload ownership rule (executor-audit finding):** every element of an anchor's payload must be host-owned or a dedicated right-sized allocation — NEVER a view/clone retaining a larger transient allocation. Today `pending_target_hidden` is an `MlxBuffer` captured by cloning a view whose parent is the prefill residual allocation (engine_qwen35.rs:3406, capture sites :3972/:4759): the logical row is ~20 KiB, but the clone retains the ~40 MiB (2,048-row) / ~80 MiB (4,096-row) parent Metal allocation — one per anchor. Capture must copy the row into a dedicated `[1, H]` allocation or host memory. Required regression: after capture, assert no chunk-sized parent allocation remains retained by any anchor (allocation-accounting check).
- Budget: `HybridKvSlotAnchor::total_bytes()` (kv_cache.rs:1004) undercounts — it omits prompt tokens, the vocab-sized `prefill_logits`, the spec hidden row owned by `Qwen35PromptAnchor` (engine.rs:16865-16871, spec boundary struct engine_qwen35.rs:3404-3407), and the store's retained `Vec` control allocation. Account **all owned bytes** as a separate reclaimable `anchor_owned_bytes` line surfaced to admission — NOT added to scheduler high-water, which is deliberately monotonic allocation accounting and is not proof that overwrite-backed Metal pages are demand-resident. Host-owned evictable bytes charged there would never return. **K counts committed anchors only; every capture preflight charges the slot's committed payload plus one pending payload against the live aggregate.** Preflight fail-closed: a capture that would exceed the immutable worker-lifetime anchor grant is skipped, never partially taken.
- The budget is aggregate across all slots, not `N × a per-slot constant`. `K_effective = min(4, floor(aggregate_budget / (N × anchor_bytes(model))))` is committed-depth capacity. The separate `simultaneous_pending_capacity_slots = floor((aggregate_budget - N × K_effective × anchor_bytes) / anchor_bytes)` makes concurrent pending availability explicit. If that value is below N, later simultaneous captures may skip fail-closed even though every slot can retain K committed anchors. Both values, N, aggregate owned bytes, peak bytes, skips, and the immutable grant are production telemetry; no N=16 receipt may claim full depth or full pending concurrency without those gauges proving it.
- Rev 97 makes “aggregate” literal for every supported generative family.
  Family-native prospective-byte functions must match the payload that would
  be captured, and admission runs before snapshot allocation/copy. Payloads
  temporarily owned outside `AnchorStore` during a bounded or rectangular
  transaction remain charged to the same immutable worker ledger, including
  extracted lanes. Replacing one transient boundary releases only that
  boundary's charge before admitting its successor. Park/reactivate may
  rebuild controls and sessions, but it may not discover a new grant.
- Per-model anchor cost (computed from allocation code, never doc comments):

  | Model | recurrent+conv state | ≈ total w/ logits+tokens+hidden | K=4 × 4 slots | K=4 × 8 slots | K=4 × 16 slots |
  |---|---|---|---|---|---|
  | Qwen3.6-35B-A3B (30 DeltaNet layers) | 62.8 MiB | ≈63.5 MiB | ≈1.02 GiB | ≈2.03 GiB | ≈4.06 GiB |
  | Qwen3.8-27B (48 layers) | 149.6 MiB | ≈150.3 MiB | ≈2.35 GiB | ≈4.70 GiB | ≈9.39 GiB |

  DeepSeek4's mature-boundary snapshot owns 49,299,456 bytes (47.015625 MiB)
  before token identity and small control allocations: 43 × 128 × 512 BF16
  circular-window rows, 21 ratio-4 layers of main+indexer F32 compressor state,
  and 20 ratio-128 layers of main F32 compressor state. Prompt identity adds
  exactly `4 × token_count` bytes per anchor. Thus K=4 base snapshot payload is
  ≈0.735/1.469/2.938 GiB at N=4/8/16; at a 100K-token boundary those become
  ≈0.741/1.481/2.962 GiB. Gemma4 uses the same exact family-owned accounting
  discipline rather than this DeepSeek-specific formula.
- Idle conservation invariant: at scheduler idle, first prove every physical target cursor equals its retained ledger and every retained spec boundary has matching target+MTP cursors. Then audit the scheduler-accounted cursor bytes, monotonic allocation slack, KV free bytes, exact host-owned anchor bytes, and anchor free bytes against their two grants. The equalities are not substitutes for the cursor comparisons.

**A.5 — Speculation interplay (LIVE requirements — corrected after the executor audit and selected-boundary spike).** Slot-aware speculative rollback is live via its own transactional functions — `rollback_slot_mtp_transaction` and `rollback_slot_target_transaction`, with fail-closed slot reset if rollback itself fails. H38 still pins `rollback_la_to` out of the slot-aware worker; that does not prohibit a separately proven stable-boundary capture using the same grow-only storage. The rules below bind NOW and are co-designed with the speculation lane:
1. Anchor capture and restore must be sequenced strictly outside an open MTP/target rollback transaction — never observe mid-transaction cursors. Prefill-time anchors copy bytes OUT of live buffers and are safe once that sequencing holds.
2. After any restore, the target cursor must equal the anchor's `token_count`.
   If the restored anchor carries `Qwen35SpecPrefixBoundary`, the MTP cursor
   must also equal that token count before speculation can resume. A
   target-only anchor may retain an independent MTP cursor only while
   speculation is disabled until exact catch-up or reset; target/MTP equality
   is not an unconditional property of every anchor.
3. A decode-time anchor (not in this ADR's scope) may only snapshot after accept/rollback settles — never from optimistic post-verify state; any capture sharing the LA storage must extract its rows before `clear_la_capture`/re-arm.
4. A stable prompt boundary inside a larger scheduler slice must preserve the
   two authoritative segments' exact operator shapes, cache representations,
   route selectors, output-head rows, and MTP batches. One enclosing forward
   is rejected: source review produced concrete counterexamples at prefix/
   suffix widths 33/1 and 33/8, cold TQ cache, recurrent/chunk DeltaNet width
   65/128, and MoE width 33/4 with `top_k=8`. A sampled output-hash sweep cannot
   prove equality when those operation routes differ. The smaller scheduler-
   coalescing spike invoked the two existing exact forwards within one bounded
   advance and was falsified on hardware: 15/15 normalized responses matched,
   while OFF-to-ON medians at suffix widths 0/2/4/8/12 changed by
   +0.18%/+1.18%/+0.69%/-0.18%/-0.57%. It did not touch meaningful cost and
   its runtime code was removed. The active hypothesis is a block-segmented
   per-layer API with separate prefix/suffix state and terminal boundary-state
   copies; every operator must still receive its control-path shape and cache
   representation. It may not land without byte-level state, logits,
   next-token, failure-atomicity, multi-slot, and performance gates.

The full 32-token SerialFifo capture window is ≈1.96 GiB for the documented
Qwen3.6 shape (30 layers × ≈67 MiB), four times the deleted stale comment.
Lane A does not allocate that window per slot: the authoritative split forward
already ends at the stable boundary and snapshots its terminal state. A future
segmented implementation may use a one-row terminal-copy buffer, but the
falsified enclosing-forward path cannot justify it. Physical storage may
remain larger after an earlier legitimate use; provenance and active depth,
not allocation capacity, define snapshot authority.

**A.6 — Family parity (REQUIRED phases per the scope directive, same invariants):** gemma4 (`Gemma4PromptAnchor` is structurally identical; retire serial-path `live_prefix_tokens` special-casing where subsumed), then deepseek4 (its two-anchor pending/committed pair generalizes to the store; wire `Deepseek4CacheSnapshot::resident_bytes()` — currently zero callers — into the same accounting). The ADR does not reach Implemented until every supported serving family carries the anchor store.

Gemma4 model-free milestone (rev 7): COMPLETE. The single
`Option<Gemma4PromptAnchor>` is replaced by `AnchorStore<Gemma4PromptAnchor>`
without changing Gemma's existing dense/MoE execution or mixed-prefill row
aggregation. Checkpoints remain the family-native `GemmaHybridSlotAnchor` plus
the rendered token prefix and vision fingerprint. Restore now preflights every
layer, optional buffer, destination span, and cursor before the first write;
single- and multi-slot failures clear the whole store and hard-reset rather
than trying a shallower checkpoint. Captures stage at the stable boundary but
remain invisible until terminal retained-token publication; cancellation
discards pending state while preserving only physically reachable committed
ancestors. Production metrics expose capture cost/skips, hit/miss/tokens saved,
descendant pruning, eviction/cancellation/lineage clears, per-slot and aggregate
peak bytes, configured slots, and the immutable aggregate grant.

Proof at base `2c6bcb61` (model-free/synthetic, no model artifact):
`cargo check --locked --bin hf2q --no-default-features`; 24/24 focused generic
and Qwen wrapper tests via `cargo test --locked --bin hf2q anchor_store
--no-default-features`; and 5/5 Gemma tests via `cargo test --locked --bin hf2q
gemma_anchor --no-default-features`. The battery includes A→B→C rewind and old
descendant rejection, request cancellation with an unchanged committed set,
failed-restore full-store invalidation, exact family-owned byte conservation,
pending invisibility until terminal publication, image-fingerprint equality,
and a two-layer late-layout fault proving no earlier byte or cursor mutates
before all-layer preflight succeeds.

Gemma4 real-model milestone (rev 22): COMPLETE. The canonical four-slot server
ran the 20,576,631,488-byte
`gemma4-ara-2pass-APEX-Q5_K_M.gguf` on an M5 Max from the source tree committed
as `c0be27fd`. The artifact is the 30-layer, 128-expert/8-active MoE shape and
reports a Q6_K tied embedding/output head. Every ordinary matrix remained
file-backed (20,560,475,768 bytes, zero ordinary anonymous matrix bytes); the
only named anonymous state was 337,920 bytes of derived router data. The
canonical server reached synchronous warmup in 711 ms after the argmax-params
dtype correction described in the revision ledger above.

The reusable family-neutral gate
`scripts/test_family_slot_anchor_lineage.sh` passed A→B→C, native equality,
rewind to X, stale-C rejection, four overlapping clients, semantic SSE
cancellation, one-shot failed restore (HTTP 500), hard-reset cold recovery,
and rebuilt semantic-equal reuse. It recorded 10 restore hits, one classified
restore miss, four descendant prunes, one cancellation, 0 cached tokens after
the fault, and 58 cached tokens after rebuilding. Effective committed depth,
simultaneous pending capacity, and configured slots were all four; the exact
1,265,327,240-byte aggregate peak stayed within the 8-GiB grant. One sibling
inherited the cancelled owner's slot-local boundary and two independently
admitted siblings completed cold, matching Lane A's slot-local contract. The
summary is
`/opt/hf2q-evidence/gemma4-anchor-rev22-spike-v2/summary.json` (SHA-256
`428df87be50816cffbd5d0d755ca1964d9f27940269039bc11cd4f33f61da2b3`).
Dense/MoE execution variants share this state engine and payload contract;
their model-free coverage remains authoritative rather than multiplying the
semantic gate into another artifact Cartesian product.

DeepSeek4 model-free milestone (rev 8): COMPLETE. Its previous single
committed plus single pending recovery pair is now the same model-neutral
`AnchorStore<Deepseek4PromptAnchor>` state machine, while the payload and match
rules remain family-native. Anchors own compact circular-window tails and
recurrent compressor state, deliberately do not own logits, and therefore
match only strict prompt prefixes; exact live-prompt equality still uses the
live ledger and live logits without replay. Selection is deepest-surviving
linear lineage. Restoring A prunes every deeper anchor before divergent writes.
Cold reset, poison, or any failed restore clears the entire store before a hard
reset; restore itself validates every layer, shape, dtype, destination span,
and cursor before the first copy or cursor mutation.

Cache growth no longer rewrites one privileged snapshot. Every committed and
pending snapshot is preflighted against the source and destination plans, the
live cache migrates once, and immutable snapshots from any prefix-compatible
growth ancestor remain restorable. This avoids both stale anchors and an
O(anchor-count) device-copy tax during session growth. Exact aggregate
accounting charges the preallocated store controls, every committed and pending
snapshot allocation, cloned cache-plan controls, and `4 × prompt_tokens`; an
immutable worker grant covers all configured stores and reports effective K
plus simultaneous-pending capacity. DeepSeek4 has no live speculative-prefix
payload at this revision, so the family adds no parallel speculative boundary
state; any future DeepSeek speculation must extend this same anchor payload.

Proof on base `234fb394` (model-free/synthetic, no model artifact): 20/20
DeepSeek cache tests, including a late-layer restore fault that leaves every
earlier byte and cursor unchanged and multi-anchor capacity growth; 19/19
DeepSeek serving tests, including A→B→C rewind, pending invisibility,
cancellation, reset/poison invalidation, failed-restore hard-reset+clear,
two-slot family isolation, native equality, and aggregate-byte conservation;
and `cargo check --locked --all-targets --no-default-features`.

DeepSeek4 real-model milestone (rev 23): COMPLETE. The canonical four-slot
server ran the 107,431,343,168-byte
`DeepSeek-V4-Flash-0731-agentic-q2.gguf` (SHA-256
`936a97e68fe1a04185df149fcb833c3e1462ca5923fbf4ef3e7296bd78c7ad0d`)
on an M5 Max. The artifact is the
43-layer, 256-expert/6-active DeepSeek-V4-Flash-0731 shape. All 387,840 rows
across its three `tid2eid` tables contained six distinct, in-range expert IDs,
so the validated Q2_K grouped expert route was used; repeat-bearing tables
remain exact through the MV/flattened-down fallback. Weight residency was
107,430,170,972 logical bytes: 107,424,498,012 file-backed bytes and only
5,672,960 anonymous bytes for quantized elementwise F32-consumer state, across
two mapped segments. Matrix weights were neither dequantized nor requantized.
After the independently tracked identity scan, residency setup took 141 ms and
synchronous warmup took 15.147 s.

The family-neutral product gate passed unary/SSE strict-prefix reuse, native
equality, A→B→C and rewind, stale-descendant rejection, four simultaneous
clients, semantic cancellation, injected restore failure with HTTP 500,
hard-reset cold recovery, and rebuilt semantic-equal reuse. It recorded 426
cached tokens from A to B, 441 on equality, 12 restore hits, four classified
misses, four descendant prunes, one cancellation, zero cached tokens after the
fault, and 37 after rebuilding. Configured slots, effective committed depth,
and simultaneous pending capacity were all four; the 89,380,036-byte aggregate
peak stayed within the 4,458,690,969-byte grant. The cold 434-token prompt
prefilled at 342.17 tokens/s and decoded at 34.60 tokens/s on this exact
artifact and run. The summary is
`/opt/hf2q-evidence/deepseek4-anchor-rev23-spike/summary.json` (SHA-256
`429b9b4088fe124b1299d016128802fcbce495e6b10328da6c8299d9ca44e276`);
the runtime binary SHA-256 was
`787729306ac15ee4bc29660aa4e17cbd02325a46876f7521d0091d02123f1aa2`.
This closes Lane A's supported-family hardware cell. DeepSeek cache-growth
capacity remains covered by its separate cache gate rather than multiplying
this semantic-state receipt.

**A.7 — Open hypotheses: the spikes that decide them** (not "deferred" — each is a live question whose data collection is already scheduled or cheap):
- *Cross-slot / restart-surviving registry tier*. Hypothesis: foreign-slot landings and restart warm-up are frequent enough on the real workload to justify a shared CoW prefix store. Deciding data: A.8 telemetry — slot-affinity foreign-landing counts and restart-cold counts over production use. If confirmed → its own ADR (needs a slot-parameterized `restore_partial` — the current one is copy-owning and rewrites every sequence cursor, kv_cache.rs:3435, :3493 — plus an ownership answer to the `b44b92ed` tenant-isolation pin; FreeToken's donate-not-copy/CoW/dual-currency eviction is that ADR's design vocabulary). Meanwhile the SerialFifo `LcpRegistry` + disk hydrate remains the restart-hydrate tier.
- *Dense/stride state capture* (finer-than-boundary anchors). Hypothesis: harness edits land off semantic boundaries often enough that boundary anchors miss real reuse. Deciding data: A.8 divergence-position histograms (an off-boundary edit degrades to cold, never to wrong — so the data can be gathered safely in production). Decision ladder if confirmed: first another semantic oracle (verified tool-call opener, FreeToken's `--enable-special-token-ckpt` analog — cheaper, targeted), then a stride-mode capture kernel under its own ADR and byte budget. Note (corrected FreeToken characterization): FreeToken itself records only the deepest crossed 64-boundary per forward, not every boundary — dense-stride economics were never validated even there.
- *Decode-time anchors*. Hypothesis: divergence points inside generated spans (not just prompt boundaries) carry meaningful reuse. Deciding data: the same histograms, split by prompt-span vs generated-span divergence. If confirmed → A.5 rule 3 already defines the safe capture point.
- *SerialFifo recovery-arena reclaim* (~1.96 GiB). Hypothesis: once the anchor store lands, the 32-token recovery-capture path is redundant for boundary edits. Deciding data: post-Phase-2 recovery-capture hit rate vs anchor coverage. If confirmed → cap or retire the arena in a follow-on.

**A.8 — Telemetry before policy** (ships with A.1): per restore attempt —
family, slot when applicable, cause, terminal outcome, attempted/hit depth,
matched-old-tail divergence position and distance, tokens saved, descendant-
prune count, whether pending state was actually discarded, publication/eviction
disposition, capture ms, and true per-slot peak committed+pending bytes. One
event is emitted only after restore+prune succeeds or fail-closed cleanup
finishes. A stable Gemma cohort emits one independently classified event per
attempted lane; one lane's reset failure cannot relabel or suppress its peers.
No-match and unpublished transient-rollback cases use explicit non-applicable
fields instead of inventing a publication disposition. This data decides every
open policy refinement above.

**Lane A gates (all fail-closed):**
1. Byte-identity: the model-free battery compares anchor-restore-plus-suffix
   prefill with cold full prefill at every anchor depth and at slice-edge plus
   mid-slice boundaries. Real-model gates exercise each distinct transaction
   route once; codecs and slot widths that share that state route are covered
   by the native-format/physical-width matrix rather than duplicated here.
2. The A.2 lineage regression (A→B→C / rewind / old-C-must-not-restore).
3. Cancellation: cancel mid-prefill after ≥2 anchors installed; committed list must equal the pre-request list.
4. `scripts/test_qwen35_slot_anchor_divergence.sh`: explicit
   `--scheduler inflight-batched` plus an exact configurable `--max-slots`,
   truly concurrent clients, equality hits, divergent rewrites, cancellation,
   admission isolation, speculative-state carry, aggregate-capacity receipts,
   stale-descendant rejection, and a one-shot post-admission failure after a
   successful non-empty GPU prefill slice. That fault is centrally parsed,
   requires `HF2Q_UNSAFE_EXPERIMENTS=1`, fires before scheduler publication,
   and the gate requires its 5xx plus exact counter delta, worker readiness,
   cold retry of the affected unique boundary, and reuse after rebuilding it.
   An oversized pre-admission 4xx remains admission-isolation evidence only.
   The script exits nonzero
   on every miss and refuses to run when the listener process arguments do not
   prove the required scheduler shape. (`bench_lcp_resume_speedup.sh` is NOT a
   gate for this feature: it drives the stride registry, issues its "4-worker"
   load sequentially, does not select the slot-aware scheduler, and exits 0 on
   a failed speedup — bench_lcp_resume_speedup.sh:303, :442.
   `test_agentic_cache_lifecycle.sh` covers cancellation/isolation, not
   multi-depth lineage; keep it, extend nothing into it.)
5. Perf acceptance: on the divergent-edit scenario, TTFT strictly better than cold; on append-only scenarios, byte-stable and within noise of today. Quiet box, receipts.
6. `cargo test --bin hf2q` (never `--lib`), 40-module GPU lock intact.

### Lane B — Mixed-phase cooperative prefill (deepseek4)

**B.0 — Coherence spike, first, in its own branch (Kimi M1 — this gate decides the lane).** Mixed cooperative execution has never run: cohort commit/poison concurrent with live decode cursors on peer slots. Before any policy work: enumerate exactly which per-slot state the cooperative transaction touches (`Deepseek4CooperativePrefillPlan` path, all-or-poison commit via `publish_prefill_cohort_after_gate` — verifier_forward.rs:76, the cooperative-PREFILL publisher; `publish_verifier_cohort_after_gate` in decode_cohort.rs:12 is the DECODE publisher, which a prior draft misnamed here; direct session-cache borrow engine_deepseek4.rs:649-683). The spike must exercise BOTH the commit and poison paths of the prefill publisher, and their interplay with the decode publisher during a Mixed step, prove decode-lane KV append cursors and compressor accumulators are untouched by commit AND by poison, and land a byte-identity test shaped *cohort-prefill + concurrent decode step* (not cohort-prefill alone). **Abort criterion:** if the spike shows cohort commit touches decode-lane state, the `engine.rs:9716-9731` bypass is load-bearing for correctness, Lane B becomes a scheduler redesign, and it exits this ADR to its own.

**B.0 deciding-spike verdict (2026-08-22): PASS; B.1 may proceed.** The source-grounded ownership ledger is:

- Planning is read-only. `plan_deepseek4_prefill_cohort` selects only FIFO handles whose installed work is `Prefill`; `prepare_cooperative_prefill` validates the selected cache cursor/token ledger/live-logit state and reserves selected token-ledger capacity without publishing tokens.
- `supervised_verifier_prefill_cohort` forms mutable cache references from only the sorted selected slot indices. The uncommitted transaction can write those selected caches' circular/compressed attention storage, indexer rows, and the four F32 compressor accumulators (`main_kv_state`, `main_score_state`, `indexer_kv_state`, `indexer_score_state`). Its combined row state, attention buffer, FFN state, and `mm_id` scratch are transaction-local model buffers, not slot state.
- `publish_prefill_cohort_after_gate` prevalidates and then advances only the selected caches' `next_position`. A prevalidation or supervisor-gate failure sets `poisoned` only on those selected caches; physical writes are deliberately not rolled back and remain invisible because a poisoned cache must reset/replay. It does not receive a peer decode cache reference.
- After a successful cache publication, serving mutates only each selected prefill lane's pending recovery anchor (when the boundary is crossed), committed-token ledger, `Deepseek4PrefillState` cursor/progress, and scheduler prefill accounting. Final logits remain on the serial completion path. The decode publisher independently validates and advances only its four selected decode cursors.

The executable proof is `mixed_prefill_commit_preserves_decode_lane_cursor_and_cache_bytes` plus `mixed_prefill_poison_preserves_decode_lane_cursor_and_cache_bytes` in `src/inference/models/deepseek4/mixed_coherence_tests.rs`. Both build six independent warm caches (two aligned prefill lanes and four live decode lanes), execute the real decode publisher first to mirror a Mixed step, then execute cooperative-prefill publication. The first accepts the prefill commit; the second rejects its supervisor gate and poisons the two prefill caches. Both byte-compare every decode lane's complete attention/KV backing, indexer storage, and all four compressor accumulators, and require its cursor to remain exactly at the one position published by its own decode step. Result: 2 passed, 0 failed. The bypass is not load-bearing for cache-state isolation; B.1 remains responsible for scheduler/SSE latency and product-performance contracts.

**B.1 — Mixed cohort policy (only after B.0 passes).** Treat this as a new scheduler policy, not the removal of one bypass: cohort planning under a runnable decode must honor a per-lane row cap AND the aggregate cap while preserving FIFO-prefix compatibility, identical-plan/reply-class requirements, and recovery-tail behavior (`plan_deepseek4_prefill_cohort`; `Deepseek4PrefillState::plan_cooperative_prefill`; `MIN_MATRIX_APPEND_TOKENS`; `RECOVERY_TAIL_TOKENS`).
- Production is fixed at **4×128** (halves F payments per aggregate progress
  while keeping GPU occupancy nearest today's serial Mixed slice). The pure
  helper accepts 256 for a deciding experiment, but there is no live 4×256
  selector; promoting it requires a measured policy implementation and the
  same contracts. Projected walls for the canonical four ~3,520-token warm-
  suffix workload remain serial ≈75 s, 4×256 ≈31 s, and 4×128 ≈46 s until
  hardware replaces the projection.
- Dual latency contract, both fail-closed: (i) scheduler decode-visit gap bound; (ii) client-visible **semantic SSE gap** bound per active decoder. Numbers to be fixed in the spike branch from measured baselines, recorded in this ADR at execution time.
- New required workload test: four prefills + active streaming decoders, measuring decoder starvation behind Mixed prefill — the existing B4 artifact proof is a pure 132-step decode comparison after prefixes are installed (real_artifact_decode_cohort_tests.rs:309) and does not cover this.

**B.1 implementation-candidate evidence (2026-08-22; hardware acceptance still OPEN).** `deepseek4_mixed_work_budget` now supplies two independent Mixed limits whenever decode is runnable: the existing two-window serial limit and a 128-row cooperative per-lane limit. `advance_deepseek4_prefill_quantum` attempts the cooperative plan even when the serial fallback is bounded; `plan_deepseek4_prefill_cohort` intersects the per-lane limit with the unchanged `MAX_COOPERATIVE_PREFILL_ROWS = 2_048` aggregate ceiling and accepts only the registered `{128, 256}` policy values. The scheduler-selected oldest prefill remains the cooperative primary, so the existing FIFO-prefix, identical-plan, reply-class, cold-wave-width, and client-open checks remain authoritative. The round-robin choice is only the serial fallback. A selected recovery tail still suppresses decode for its bounded replay and bypasses cohort execution, preserving the established cursor-alignment rule.

The execution still makes one `supervised_verifier_prefill_cohort` call over the combined row set and uses the existing `forward_ffn_rows`/`mm_id` path; B.1 adds no per-expert loop or dispatch. Pure-prefill calls pass no lane cap and therefore retain their previous aggregate-cap row allocation. No product ceiling or `MAX_COOPERATIVE_PREFILL_ROWS` changed.

Two receipts are now bound to production events for every successful SlotAware DeepSeek request: `scheduler_decode_max_gap_ms` is observed once at actual decode-batch entry (the first gap begins when decode becomes runnable), while `semantic_sse_max_gap_ms` is observed only after an actual content/reasoning/tool delta is accepted by the SSE channel. Empty role frames and token-generation proxies cannot satisfy the semantic receipt. Both carry observation counts and first-event timestamps under the same request id; the model-free bound verifier rejects missing observations, a gap above its bound, or a non-monotonic trace. The hardware spike must set the numeric bounds from quiet-box baselines and apply the same fail-closed rules before either latency contract can be accepted.

Model-free proof at this candidate: `deepseek_mixed_cohort_rows_fail_closed_on_lane_and_aggregate_caps` pins the 2,048 aggregate ceiling, shipping 4×128 shape, optional 4×256 shape, pure-prefill allocation, and rejection of sub-window/unregistered caps. `deepseek4_four_prefills_and_live_decoders_visit_every_mixed_turn` drives an eight-slot scheduler for five Mixed turns and proves the same four decoders are visited every turn in stable FIFO order while the same four-prefill FIFO prefix advances by exactly 128 rows/lane (512 aggregate). `latency_gap_receipt_is_measurable_and_fails_closed` proves missing-observation, over-bound, and non-monotonic traces reject; `semantic_sse_receipt_records_first_event_and_maximum_gap` proves the actual semantic send path records every event and the exact maximum gap. Focused result: 138 passed, 0 failed, 8 artifact/hardware tests ignored. This is implementation evidence, not the required real-artifact performance/parity, semantic-SSE ceiling, thermal, or peak-RSS evidence.

**B.1 hardware falsifier (2026-08-26; no sample admitted).** The first exact
ABBA invocation verified the 107.43-GB artifact and reached the real eight-slot
OFF warmup, but `extract_request_ids` then passed a literal line-continuation
backslash to Perl and both budget-specific regular expressions failed to
compile. The producer rejected the warmup before thermal settlement or any
performance sample. Removing the embedded backslash is the complete parser
fix; the model-free shell contract now extracts and executes that exact
function over representative 256-token, 8-token, near-match, and unrelated
log rows. Hardware acceptance remains open pending a fresh receipt directory.

The next exact invocation at source `d8ef4b28` passed that parser proof and
reached the OFF warmup, then falsified the workload/discriminator before any
sample. The four 160-token decoder requests formed a cold cohort. Although the
four long-prefill HTTP requests arrived while those decoders streamed, the
accepted cold-wave barrier intentionally retained them until the first cohort
completed; their later 4x512 cooperative transactions were the canonical pure-
prefill route, which remains enabled in both arms. Treating every cooperative
event as Mixed therefore produced a false OFF failure. Rev 88 preserves the
cold-wave policy and reforms the gate around stable per-lane decoder priming:
each measured decoder must restore nonzero cached state, every prefill must
overlap those warm decoders, and the runtime receipt labels whether a cohort
was bounded by the 128-row Mixed cap. ON requires at least one labeled 4x128
transaction; OFF requires zero. Later unbounded pure-prefill transactions are
recorded but cannot satisfy or falsify the Mixed discriminator. Hardware
acceptance remains open pending the fresh exact-lineage ABBA rerun. The same
pre-run proof audit corrected the aggregate semantic digest from a nested
per-wave verifier projection to the producer's flat per-response projection;
the exact digest now also includes the four priming responses per replica.
Rev 89 additionally binds the declared workload rather than its labels: each
decoder plan must conserve `cached + work = prompt`, use an accepted live or
recovery-anchor action, and match the client-visible cached-token multiset;
each long prefill must use a reset action with `cached = 0`, a full uncached
suffix, bounded cold work (allowing the existing recovery-checkpoint prepass),
and zero client-visible cached tokens. Runtime request identities prove all
four prefills were admitted before the first decoder completed. The receipt
mutation battery rejects cache-action, conservation, client-usage, cold-
prefill, and admission-order substitutions, while an accept-preserving pure
4x512 insertion proves that unbounded cooperation is intentionally ignored.

Rev 90 rejects the shared `process-group-v1` host-contention receipt as
insufficient performance authority. A live preflight observed roughly 600%
CPU from a three-day orphaned headless-browser tree plus unrelated analysis,
while the old name-only snapshot still emitted a valid `quiet` row because it
recognized only compilers, peer binaries, and foreign hf2q processes. No model
run or performance sample was started. `process-group-cpu-v2` adds `%cpu` to
the normalized snapshot, excludes the gate's complete process group, sums all
foreign CPU, and rejects a sample at `>= 100.0%` foreign CPU while retaining
the prior unconditional named-process rules. The six-column receipt records
that aggregate explicitly. The existing continuous 60-second quiet settle and
first-contended-sample measurement failure are unchanged. GPU-only process
attribution is not claimed: the available aggregate probes cannot distinguish
owned inference from foreign work, and privileged process-GPU sampling needs
its own overhead-calibration spike. The actual observed blocker is CPU-visible
and is now fail-closed. DeepSeek B.1 binds the v2 policy in its top workload
receipt; its mutation battery is 19/19 (18 rejected corruptions and one
accept-preserving unbounded-pure event).
The orphaned browser started on August 22, before both the rev-86 Gemma
product run and the August 25 Qwen curve, rectangular, Q5, and matched-peer
timing receipts. None of those v1 receipts can prove that its timing window
was free of generic foreign CPU. Gemma and Qwen exact state/coherence gates
and their shipped implementations are not reversed, but every affected speed
acceptance and timing-derived policy conclusion is provisional until its
lower-confidence bounds reproduce under v2. The historical raw measurements
remain evidence; they are not uncontended performance authority.

**B.2 — Family generalization (scope directive).** The earlier claim that
cross-slot aggregation was DeepSeek-only drifted from code. Gemma SlotAware
already aggregates compatible installed lanes in
`advance_gemma4_prefill_quantum` through
`forward_prefill_batched_multi_seq{,_live}` under the canonical
`HF2Q_CROSS_SLOT_ADMIT=1` launcher policy; ADR-040 records that shipment.
Qwen MoE was the first known missing family. The deciding spike therefore
measures Qwen's fixed-cost opportunity and revalidates Gemma rather than
assuming that source reachability proves coherent shipment. The checked-in
Qwen curve harness uses
cold, single-transaction HTTP requests at controlled row widths, exact
binary/model/process/request/response/thermal/contention receipts, two warmups,
an initial seven alternating sweeps, and a deterministic bootstrap fit of
`t(R)=F+cR`. A valid but inconclusive initial result permits exactly 21 sweeps
with every other contract unchanged.
Qwen opens implementation only when the lower 95% bound shows fixed-cost share
above 50% at 128 rows and projected four-lane gain above 1.10×; it is falsified
only when the upper bound is at most 25% or projected gain at most 1.05×.
Intermediate evidence is inconclusive and triggers more measurement. Gemma is
OPEN until direct OFF/ON order-balanced trials retain exact per-lane results
and the lower 95% speedup bound exceeds 1.05×; the former "existing policy"
claim was withdrawn by the rev-76 canonical-route hardware falsifiers.
The sequential curve harness cannot satisfy that distinct Gemma four-lane
OFF/ON contract.

Rev 24 Qwen gate calibration (not performance evidence): the first exact
Qwen3.6-35B-A3B run at `54ba51df` produced 45 cold, single-transaction
responses, but the verifier rejected sample one. The nominal 128-word payload
rendered to 164/165 prompt rows because the generator ignored the 36–37-token
ChatML envelope; an attached `script(1)` capture also buffered the server log,
so per-request trace slices were empty even though the final log contained all
events. Raw evidence remains at
`/opt/hf2q-evidence/adr049-b2-qwen35-moe-54ba51df-rev1`. The revised harness
subtracts the 36-token lower envelope, treats response/trace work rows as final
authority, strips optional ANSI from trace fields, and sends a cold 128-row
preflight canary that requires exactly one live production transaction before
the 60-second thermal settle.

The corrected seven-trial receipt is valid at `e7f69c36` (binary SHA-256
`0227de0c2229dc75a791e6c246f0e1d0f5e9872c5d85c4727eb9cfaed2eba164`,
model SHA-256
`f2c702182a4661d2cef573b388ff23336ce65aabb112762d1c1a24d4ba0cbc25`):
`F=82.989 ms`, `c=0.5881 ms/row`, `R²=0.99946`, fixed share at 128 rows
52.44% with 95% CI `[49.00%,55.55%]`, and projected 4×128 gain 1.648× with
95% CI `[1.581×,1.714×]`. The fixed-share lower bound is below the strict
`>50%` confirmation rule, while neither falsification rule holds, so the
terminal result is **inconclusive**, not a win. Immutable receipt:
`/opt/hf2q-evidence/adr049-b2-qwen35-moe-e7f69c36-rev2/summary.json`
(SHA-256
`ce16c84affe37e35314e080306b8c87fa4ecda9dbd53975526673be8c949a47e`).
The pre-registered precision extension is exactly 21 trials per width; it does
not change widths, alternating order, bootstrap seed, fit validity, or terminal
thresholds. That extension passed at `65495b4d` (binary SHA-256
`1a273fce7ebf1a31595d4c7ea41a92301f9d963b621e484202d7c4a0730418d8`):
`F=85.268 ms`, `c=0.5642 ms/row`, `R²=0.99948`, fixed share at 128 rows
54.14% with 95% CI `[52.58%,55.13%]`, and projected 4×128 gain 1.684×
with 95% CI `[1.651×,1.705×]`. Both strict lower-bound rules pass, so the
Qwen opportunity is **confirmed** and implementation is required. Immutable
receipt:
`/opt/hf2q-evidence/adr049-b2-qwen35-moe-65495b4d-rev3/summary.json`
(SHA-256
`19571ce1dc44fc5d5cd52f669f885253a370442bc69731e494da14b497160709`).

The implementation hypothesis is a rectangular physical batch of two through
four compatible lanes with equal row count. Sequence-major rows stay
contiguous so existing dense and MoE projections see one aggregate matrix and
`mm_id` remains one dispatch, not an expert or lane loop. DeltaNet uses its
native `n_tokens × n_seqs` axes with gathered current state and parity-correct
scatter. Full attention must preserve the scalar operator class: production
128-row fresh prefill stages the aggregate rows as an isolated native batch
and invokes the BF16 tiled dispatcher once; explicit row-to-slot and absolute-
position maps write each lane's freshly projected K/V into its own banked TQ
cache. A resumed batch is compatible only when every lane shares cursor, K
length, physical capacity, and scalar route, allowing the same tiled-resume
dispatcher with its native batch axis; otherwise the scheduler keeps those
lanes scalar. The rejected alternative widened the TQ decode-vector kernel for
128-row prefill. It changed the attention arithmetic/cache representation
relative to the scalar tiled route and was removed before publication. Stable
boundaries become explicit cohort ends so checkpoint capture does not require
intermediate-state reconstruction. Any target, MTP, supervision, validation,
or publication failure rolls every selected lane back before any scheduler
cursor advances. The DeltaNet layer spike is green at four sequences by 128
rows with bit-identical output/recurrent/conv state and an untouched unselected
peer. The tiled full-attention bridge and the complete full-attention layer are
green at widths two and four by 128 rows with bit-identical output and complete
banked TQ state. Invalid slot/cursor/capacity shapes reject before mutation. The
complete target gate is now green for native dense and quantized-MoE fixtures
at widths two/four, including exact final logits, normalized MTP input rows,
live-prefix cache bytes, an unselected warm peer, and one-token continuation.
The MoE gate retains one aggregate gate/up/down ID projection per layer. The
worker-owned all-lane target+MTP transaction and failure/publication contract
are now model-free green at rev 31. The remaining deciding gate is exact
Qwen3.6/Qwen3.8 hardware parity and direct OFF/ON performance, including
Qwen3.8 AUTO MTP, cancellation/retry, unary/SSE/tool-result continuation, and
a live decoder in Mixed.

Rev 32 closes an operational reachability defect found before that hardware
gate. Qwen enabled `HF2Q_CROSS_SLOT_ADMIT=1`, but unlike Gemma its worker did
not collect the first idle burst: if four HTTP handlers arrived across more
than one channel poll, lane zero entered scalar prefill before its peers were
installed and the rectangular transaction could never form. The kernel path
was therefore correct but timing-luck-dependent. Qwen now uses the same
bounded worker-lifetime `HF2Q_ADMIT_COALESCE_US` policy as Gemma, and the
canonical Qwen launcher supplies the existing 25 ms maximum. Collection is
limited to a compatible FIFO prefix of cold text requests whose stable
boundary is already in the exact 16–128-row rectangular shape. Response-cache
hits, reusable retained/anchor prefixes, controls, warmup, embedding, vision,
active-work admission, and policy-OFF runs do not wait. The target is bounded
by the physical slot count, channel capacity, and the route's four-lane
maximum; an incompatible arrival or priority control ends collection instead
of consuming the rest of the window. Invalid direct-runtime values warn and
disable collection. This is implementation reachability evidence, not a
performance claim; the exact OFF/ON hardware gate below must prove both that
normal concurrent HTTP traffic forms the cohort and that the bounded
collection cost is repaid.

Rev 33 records the first exact-artifact performance falsifier, its source-
grounded reformulation, and the corrected implementation candidate. On an M5
Max, four concurrent 96-row Qwen3.8 Q5_K_M lanes formed a real width-four
cohort and returned the same semantic/token hashes as the serial control, but
the first rectangular implementation took 5.62–6.29 seconds after warmup
versus 1.83–1.97 seconds for scalar admission. Disabling MTP still took 6.07
seconds, falsifying speculation as the cause. Instrumented target time was
5.46 seconds: 4.60 seconds in dense quantized FFN, 0.70 seconds in recurrent
DeltaNet, 0.15 seconds in full attention, and 0.01 seconds in the output head.
A layer-local decode-pool reset reduced little. Disabling the pooled helper's
fused gate/up route reduced the whole wave to 1.61 seconds, but splitting the
same fused operation into exact per-lane calls still took 5.72 seconds. Those
spikes falsified both aggregate row count alone and MTP as explanations.

Source review found the actual operator-identity defect. Scalar prefill calls
the arena-backed DenseQ helper, whose gate/up projections execute native
quantized matrix-matrix operations. Rectangular prefill called the pooled,
decode-oriented helper, which selected a fused mat-vector route for the same
aggregate prompt rows. The candidate now allocates one aggregate
`DenseFfnArena` and output ring and invokes the same arena-backed helper as
scalar prefill. It also drains and drops every pool-owned layer local before a
per-layer pool reset; resetting while `attn_out` remained live would have
created a future buffer-alias footgun. GGUF buffers stay in their artifact
representation throughout. No tensor is dequantized and requantized, and the
Qwen3.6 MoE `mm_id` route remains one aggregate projection dispatch rather
than a lane or expert loop.

The first full-model exact gate is green on the 19,535,701,568-byte Qwen3.8
Q5_K_M artifact SHA-256
`4b19f41c391d962882e459be3315d4e3c54079892db2848f66b78815b185156e`.
It asserts native Q5_K embedding and native Q6_K shared output head with no
expanded CPU copies, then compares four scalar lanes with one rectangular
transaction at 96 rows: final logits, every MTP input-hidden byte, complete
hybrid cache snapshot, and a subsequent ordinary token are exact. The gate is
generalized into the qualified BF16/Q4_K_M/Q5_K_M/Q6_K/Q8_0 artifact matrix;
all five cells must pass before acceptance.

The corrected dirty-tree spike produced exact OFF/ON response hashes and
provisional median four-client walls of 1.373 versus 1.838 seconds on that
Qwen3.8 Q5_K_M artifact (25.3% lower), with AUTO MTP succeeding in every ON
process. The distinct 25,043,007,488-byte Qwen3.6 35B-A3B MoE Q5_K_M artifact
SHA-256
`f2c702182a4661d2cef573b388ff23336ce65aabb112762d1c1a24d4ba0cbc25`
produced 0.417 versus 0.623 seconds (33.1% lower) with exact response hashes.
These are hypothesis-confirming spike receipts, not landing claims: the source
tree was intentionally dirty while operator variants were isolated. Final
acceptance requires a clean exact-commit, same-binary OFF/ON ABBA runner,
every qualified Qwen3.8 native format, Qwen3.8 dense and Qwen3.6 MoE shape proof,
normal HTTP width-four publication, Qwen3.8 AUTO-MTP success, single-user
coalescing ceiling, live-decode plus four-prefill workload, cancellation/
retry/rollback joins, unary/SSE/tool-result continuation, and matched pinned-
reference measurements. A speed result that omits those cells cannot close
B.2.

Rev 34 turns that acceptance list into fail-closed executable authority before
running the clean-commit measurement. The same-binary Qwen runner freezes the performance policy before the landing
measurement: five measured trials in each fresh OFF-A/ON-A/ON-B/OFF-B
process, two route warmups per process, 60 continuous seconds of nominal and
quiet settle, Fair-or-better plus quiet measurement sampled every two seconds
with no gap above five seconds, cold unique requests, a start barrier, launch
skew at most 100 ms, and actual four-client overlap. Both ON processes must
beat their neighboring OFF process and the pooled four-client median must be
at least 1.01× faster. The single-user ceiling is end-to-end client wall time,
not the server's internal TTFT field: admission coalescing happens before that
server clock starts. Pooled ON median wall may exceed OFF by at most 50 ms,
twice the immutable 25 ms worker window. This ceiling applies to the worst
matched OFF/ON request overhead across both process pairs; pooled median is
diagnostic only, consistent with Lane C's tail-latency rule. Request bytes and canonical semantic/
token response projections must be exact across paired arms. Thermal,
contention, sampled RSS, request/response/wall, per-wave log, and metric files
are sealed into each process evidence manifest. These thresholds are not
environment-tunable by the gate. Every process also proves the exact loaded
architecture from `/v1/models` and the worker-resolved admission, coalescing,
AUTO-speculation, and MTP-capability policy from one startup event; this keeps
the gate from repeating the dead-configuration failure class. AC source and
power mode are rechecked before launch, after route warmup, at both measurement
boundaries, and after shutdown. The top receipt is reopened by an independent
verifier that recomputes raw manifests, all hashes, exact requests and canonical
responses, timing and metric deltas, launch overlap, sample counts,
thermal/contention/power contracts, medians, sampled peak RSS, tail, neighbor
ratios, and verdict. Eight injected receipt/raw-evidence mutations—including
fully rebound raw timing and metric changes—must be rejected for each cell.

Every measured server and lifecycle client starts under `env -i` with the
complete runtime/workload contract pinned. The actual startup events must prove
the 48 GiB KV budget, persistence disabled, AUTO speculation, exact
cross-slot/coalescing arm, and artifact MTP capability; the earlier
`HF2Q_KV_PERSIST=0` claim was deleted because `serve` never read it. A long-run
owned caffeinate assertion is stopped before receipt sealing, its power-event
delta is independently reconstructed, and the owned thermal probe is cleaned
on every exit path. Qwen3.8 dense/MTP and Qwen3.6 MoE/no-MTP each also run a
fresh exact-artifact agentic lifecycle process outside the timed arms. That
join reopens the canonical required-tool call, tool-result continuation,
semantic SSE cancellation, queued exact-retry reuse, isolation, execution
headers, raw evidence manifest, PID/binary/model binding, shutdown, and fatal
logs. A fully rebound lifecycle-summary mutation must be rejected from the raw
responses. These are acceptance-contract changes, not new performance results;
the clean-commit matrix remains the deciding measurement.

The first clean-commit server attempt at `85ab77cc` failed before model load:
both new direct runners still supplied the removed `serve --no-vision` flag.
Current CLI source has no such option; text-only operation is selected by not
supplying `--mmproj`. The failed invocation produced no performance sample.
The runners now use the live CLI contract, and the deciding matrix moves to the
post-correction commit rather than relabeling the failed attempt.

The next attempt at `98782fff` proved the corrected serve plan but also stopped
before model load: `HOME=/var/empty` correctly scrubbed ambient configuration,
while the live server still needs a writable model-listing cache even for a
local GGUF. The gate keeps the scrubbed home and now supplies an explicit
evidence-local `--cache-dir`; the actual PID command line and every cache file
are sealed and independently checked, including absence of `--kv-persist`.

Rev 35 records the third fail-closed clean-commit falsifier and its product
correction. At exact commit `0df8a389`, Qwen3.8 loaded with AUTO speculation
and live MTP weights, and normal concurrent HTTP traffic published a genuine
four-lane rectangular cohort. The gate nevertheless stopped after the first
measured ON wave because the publication reported
`mtp_prefill=false mtp_outcome=NotRequested`. The request fixture supplied
`temperature:0, seed:42`; source inspection showed that both the ordinary
GPU-greedy predicate and the server-complete MTP predicate rejected every
explicit seed even though `sampler_pure::sample_token` returns argmax before
consulting the RNG whenever temperature is zero. The harness had therefore
disabled the path it required, but deleting the seed from the harness would
have hidden the same reachability defect for real clients.

The corrected contract treats a seed as inert under the already-required
temperature-zero/top-k-zero/top-p-one semantics. Seeded sampling remains
outside exact MTP eligibility whenever the stochastic fields select it. A
red/green server predicate regression proves seeded-greedy admission for both
ordinary GPU argmax and Qwen speculation, while an independent sampler test
proves identical token selection for no seed, seed zero, and the maximum
`u64`. The deciding ABBA fixture intentionally retains `seed:42`; Qwen3.8 must
now complete actual MTP catch-up rather than pass by fixture omission. The
failed `0df8a389` root remains falsification evidence and makes no performance
claim.

Rev 36 records two proof-environment failures after the seeded-greedy
correction reached hardware. The first `4c88641c` retry completed three Qwen3.8
processes but correctly rejected OFF-B when an unrelated hermetic hf2q release
smoke started during the calibrated measurement window; this was real host
contention, not a false positive, and the root makes no performance claim. A
fresh uncontended retry completed all four Qwen3.8 arms with exact semantic
equality and real MTP success, but its independent verifier stopped before
judging the receipt because BSD awk rejects a continuation line whose first
token is `||`. The same script had passed `bash -n` and shellcheck because
neither parses embedded awk programs.

The verifier now keeps each awk continuation operator on the preceding line.
Its power-log predicate is one shared function used by production verification
and a directly executable `--self-test`; the contract suite runs that self-test
and proves both a valid five-phase log and a corrupt phase. The second affected
timing-shape predicate uses the same portable form. Because verifier and runner
hashes are receipt authority, the complete clean-commit matrix must run again
at the post-correction commit; the measured `4c88641c` values remain
provisional evidence only. Reopening that completed receipt through the fixed
verifier then exposed a second schema defect before another model load: the
runner derived and supplied `latest_start` and `earliest_finish` to `jq` but
did not include either in the emitted wave object, while the independent
verifier correctly required both raw boundaries. The wave schema now stores
them and the contract pins both fields. A diagnostically rebound copy of the
old raw receipt was rebound diagnostically to drive the corrected verifier to
its terminal verdict before the next exact hardware run. That rebound exposed BSD awk's
string-typed substring comparison: an extracted `"84"` compared lexically as
greater than numeric 128. Row-shape validation is now also one shared
production/self-test function, explicitly coerces both parsed fields to
numbers, accepts 4×84/336, and rejects 4×84/335. After that correction the
diagnostic shadow at
`/opt/hf2q-evidence/qwen35-rectangular-verifier-shadow.DxIKJH` reached the
terminal VERIFIED verdict and rejected all eight fully rebound receipt/raw
mutations. It proves the verifier, not a clean-commit performance result; the
next exact matrix remains authoritative.

Rev 37 records the next clean-matrix environment failure without discarding
its valid cells. At exact commit `2cdb3f0c`, Qwen3.8 completed and independently
verified: pooled four-client wall improved from 1.82005 seconds to 1.36293
seconds (1.3354×), both neighboring process ratios exceeded 1.32×, exact
semantic/token equality held, seeded AUTO MTP succeeded in every ON cohort,
and worst matched single-request overhead was 37.75 ms under the immutable
50 ms ceiling. Qwen3.6 OFF-A/ON-A then measured 0.64894 versus 0.442748
seconds, and ON-B completed all five coherent MoE/no-MTP waves. After clean
shutdown, however, the fifth power-contract row was absent and the runner
stopped before sealing that process. Four earlier identical checks succeeded,
the sampled mode/code remained automatic/zero, and the same commands succeeded
immediately afterward. This localizes the failure to transient acquisition
from `system_profiler` or `pmset`, not a measured power drift; the incomplete
Qwen3.6 cell and top matrix make no acceptance claim.

Power-mode acquisition now retries parser/command acquisition at most three
fixed attempts with one-second spacing. AC loss or any successfully observed
mode/code drift still fails immediately, and an unresolved probe now emits an
explicit phase-specific error rather than returning silently. The retry count
is immutable and source-contract tested. Because the runner hash and embedded
source commit are receipt authority, the complete matrix reruns at the
post-correction commit.

Rev 38 checks in the remaining cross-family acceptance authorities before the
next model load. DeepSeek B.1 now has an immutable worker-lifetime
`HF2Q_DEEPSEEK_MIXED_COHORT` policy: default/explicit ON retains the accepted
128-row-per-lane Mixed cohort, explicit OFF skips only the Mixed cooperative
plan, and an invalid direct-runtime value fails safe to the serial Mixed path
with its resolved selection in the startup event. The canonical launcher
rejects invalid values before model load. Its exact same-binary HTTP gate runs
fresh OFF-A/ON-A/ON-B/OFF-B processes with five measured waves each, eight
physical slots, four semantic SSE decoders plus four cold 128-row prefills,
strict pooled and neighboring-process wins, exact canonical response parity,
15-second scheduler and semantic-SSE ceilings, a 60-second prefill ceiling,
the 116 GiB memory ceiling, and continuous power/thermal/contention evidence.
An independent raw-evidence verifier and ten injected mutations guard the
receipt.

Qwen B.2 now joins an exact live-decoder/four-prefill cell for both the
Qwen3.8 dense/MTP and Qwen3.6 MoE/no-MTP artifacts to the existing rectangular
ABBA and agentic-lifecycle matrix. Gemma B.2 now has eight order-balanced
OFF/ON pairs across 128/256/512-row four-lane waves, exact request/result
parity, a deterministic 10,000-sample order-stratified bootstrap, and the
pre-registered lower-95%-speedup bound above 1.05×. Both have independent
verifiers and mutation batteries. All three new model-free contract suites are
green at this revision. These are executable acceptance contracts, not new
hardware or performance results; the unchecked acceptance items below remain
unchecked until their exact clean-commit receipts pass.

Rev 39 records the first rev-38 hardware attempt without relabeling its
incomplete top matrix. At exact commit `57bd51b4`, the Qwen3.8 Q5_K_M ABBA
cell passed independent verification: pooled four-client wall improved from
1.791365 seconds to 1.351285 seconds (1.325675×), both neighboring ratios
exceeded 1.328×, exact semantic/token equality held, seeded AUTO MTP executed,
and worst matched single-user overhead was 33.885 ms under the 50 ms ceiling.
Its immutable receipt remains valid at
`/opt/hf2q-evidence/qwen35-rectangular-policy-57bd51b4/qwen38-dense/receipt.json`.

The distinct Qwen3.6 OFF-A process then completed five clean measured waves,
30 nominal settle samples, quiet contention evidence, and four identical
AC/automatic/zero phase observations, but stopped after shutdown before the
fifth power row and before sealing a process summary. This is the second
occurrence of that boundary-acquisition failure; the Qwen3.6 cell and top
matrix make no claim. The earlier correction retried Energy Mode and numeric
mode-code acquisition but left the live AC-source command as a one-shot
early-match pipeline. The cross-family gates now capture the complete
`pmset -g batt` output, parse exactly one public AC-or-battery banner, and
retry unresolved command/parser acquisition at most three times while still
rejecting an observed battery source immediately. The shared parser has AC,
battery, missing, and duplicate-banner canaries; the Qwen rectangular/Mixed,
Gemma B.2, and DeepSeek B.1 contracts forbid the former early-match probe.
Because source and runner hashes are receipt authority, the complete matrix
reruns at the post-correction commit.

Rev 40 records the next exact attempt and closes a separate hermetic-client
defect. At `fa2f33ee`, both post-power-correction throughput cells passed:
Qwen3.8 improved from 1.800775 seconds to 1.358030 seconds (1.326020×;
neighbor ratios 1.3215×/1.3264×; 35.320 ms worst single-user overhead), and
Qwen3.6 MoE improved from 0.632006 seconds to 0.438724 seconds (1.440555×;
neighbor ratios 1.4473×/1.4344×; 31.205 ms worst single-user overhead).
Both independent verifiers proved exact semantic/token equality, all process
power rows, quiet thermal/contention contracts, and the intended MTP-capable
versus no-MTP architecture behavior.

The joined matrix then stopped before its first lifecycle request. The
lifecycle runner correctly scrubbed the client environment to
`PATH=/usr/bin:/bin:/usr/sbin:/sbin`, while the shared client still required
Homebrew `rg` for three simple fixed/anchored line checks. The model loaded and
shut down cleanly, but the client exited 2 with `missing required command: rg`;
there is no lifecycle or top-matrix claim. Those checks now use system
`grep`, the command contract no longer names `rg`, and a direct scrubbed-PATH
canary proves command preflight reaches the intended HTTP failure instead of a
missing-tool failure. The complete joined matrix reruns because lifecycle,
matrix, source, and binary identities are one receipt contract.

Rev 41 records the exact `822312b6` rerun and reformulates its lifecycle
failure from the hardware evidence. Both throughput cells passed again:
Qwen3.8 measured 1.324851× (neighbor ratios 1.3231×/1.3250×, 33.155 ms
worst matched single-user overhead), and Qwen3.6 MoE measured 1.445822×
(neighbor ratios 1.4353×/1.4511×, 32.187 ms worst matched single-user
overhead), with exact semantic/token equality. Their independently reopened
receipts are under
`/opt/hf2q-evidence/qwen35-rectangular-policy-822312b6/`.

The Qwen3.8 lifecycle base then completed a 137,589-token cold prompt and a
valid required tool call. Its seed continuation hit the retained anchor,
reported 137,584 cached tokens, and prefetched only the 171-token suffix, but
the request spent its unbounded 64-token completion entirely in native
reasoning and therefore failed the exact `CACHE_SEED_READY` assertion. This
was a request-contract defect, not a state-reuse or model-coherence defect:
the launcher value `--default-thinking-token-budget 0` removes the configured
ceiling; it does not disable a thinking-capable template. Required Qwen tool
calls synthesize a safe answer reserve, while ordinary `auto` calls do not.

Two real-artifact spikes decided the replacement contract. An explicit
request-local `thinking_token_budget=16` produced the exact seed and isolation
answers with a tokenizer-derived forced close, and the same ceiling preserved
a valid required `lifecycle_probe` call. The shared lifecycle fixture therefore
keeps the base request on the distinct synthesized required-tool reserve and
accepts an opt-in continuation budget for seed, active SSE, inherited sibling,
and isolation requests. Qwen3.6/Qwen3.8 callers pin 16; Gemma4 and DeepSeek4
leave the option absent because their family-native reasoning contracts differ.
The summary, raw-request verifier, and mutations bind that distinction. The
complete joined matrix reruns at the post-correction exact source/binary
identity; the incomplete lifecycle and top matrix still make no acceptance
claim.

Rev 42 records the next exact-candidate attempt without promoting its partial
evidence. At `2a764246`, the Qwen3.8 throughput cell passed independent
verification at 1.332332× pooled four-client speedup (neighbor ratios
1.3401×/1.3166×), exact semantic/token equality, and 40.435 ms worst matched
single-user overhead. The Qwen3.6 OFF-A process then completed all five single
and five four-slot measurements with nominal thermals, quiet contention, and
the complete through-measurement power record, but exited before publishing
the after-shutdown row or process summary. As required, neither that Qwen3.6
arm nor the joined matrix makes an acceptance claim.

An exact-source traced OFF-A reproduction against the same binary and Qwen3.6
artifact passed every measurement, post-measurement identity/fatal-log/process
binding check, graceful shutdown, after-shutdown power observation, evidence
manifest, and process-summary publication. That spike falsifies a persistent
runtime or harness defect; the unsealed exit is retained as transient rejected
evidence rather than rationalized into a result or a speculative code change.
The complete joined matrix reruns from a fresh evidence root at the next exact
source/binary identity.

Rev 43 records the full exact `6ba4c37b` attempt and its deciding correction.
Both rectangular throughput cells passed independent verification with exact
semantic/token equality: Qwen3.8 measured 1.332338× pooled four-client
speedup (neighbor ratios 1.3273×/1.3361×; 31.364 ms worst matched single-user
overhead), while Qwen3.6 measured 1.452518× (neighbor ratios
1.4637×/1.4433×; 29.161 ms worst matched overhead). Qwen3.8 then passed the
complete lifecycle gate: its 134,787-token cold base emitted the required
tool call, the seed reused 134,782 tokens, cancellation emitted no false SSE
terminal, the queued retry reused 134,947 tokens, and isolation stayed cold
and returned exact `ISOLATION_OK`.

Qwen3.6 passed the same base, seed, active-SSE, cancellation, and queued-retry
sequence. Its seed reused 134,744 tokens and its retry reused 134,909. The
unrelated request was correctly cold (`cached_tokens=0`) and leaked no prior
sentinel, but the budget-16 forced close exposed a 41-token visible meta-answer
prefix and exhausted `max_tokens=64` with `finish_reason=length` instead of
returning exact `ISOLATION_OK`. Widening the completion or reasoning budget is
falsified: neither can remove an already-invalid visible prefix, and forced-
close coverage remains live in the passing seed and sibling responses, both
of which recorded 23 reasoning tokens.

Fresh exact-artifact spikes against both Qwen3.6 and Qwen3.8 decided the
replacement: isolation alone sets `hf2q_enable_thinking=false`, carries no
reasoning budget, and carries no `chat_template_kwargs` that could override
the top-level flag. Both shapes returned exact `ISOLATION_OK`,
`finish_reason=stop`, no reasoning trace, and zero cold cached tokens. The
seed, active, and inherited sibling retain the request-local continuation
budget of 16; the base still exercises the distinct synthesized required-tool
reserve. The lifecycle summary advances to schema 3 so the renamed
`continuation_thinking_token_budget` and unrelated-conversation thinking mode
cannot be confused with the superseded all-continuations contract. Independent
raw-request/response checks and sixteen rebound mutations bind the new shape.
Gemma4 and DeepSeek4 keep their family-native fixture defaults. The Qwen3.6
lifecycle and joined matrix remain unaccepted until one fresh exact candidate
passes both lifecycle cells and the Mixed cells.

Rev 44 records the fresh exact `25f498c5` joined attempt and the deciding
Qwen3.8 Mixed-coherence falsifier without promoting its rejected top matrix.
Both rectangular throughput cells again passed independent verification with
exact OFF/ON response parity: Qwen3.8 measured 1.326704× pooled speedup
(neighbor ratios 1.3248×/1.3310×; 34.874 ms worst matched single-request
overhead) and Qwen3.6 measured 1.445323× (neighbor ratios
1.4505×/1.4365×; 33.447 ms worst overhead). Both strengthened lifecycle
cells passed from the same binary. Qwen3.8 and Qwen3.6 each completed the
137k-token cold base, reused nearly the complete stable prefix on seed and
queued retry, retained forced-close coverage on continuation requests, kept
the unrelated request cold, and returned exact non-thinking `ISOLATION_OK`.

The Qwen3.8 Mixed ABBA then correctly failed before sealing a receipt because
its 512-token open-ended decoder was not byte-identical between paired AUTO
arms. All twenty short prefill responses remained exact `OK`; only the long
decoder forked, first after 1,150 joined output bytes in replica A. The
provisional prefill direction was beneficial but is not accepted performance
evidence: pooled OFF/ON medians were 2,208.16/1,928.62 ms (1.144943×), with
both neighboring ratios above one.

Same-policy evidence rejects rectangular prefill as the causal label for that
failure. OFF-A and OFF-B also differed on the identical seeded trial-2
request; ON-A and ON-B differed on trials 4 and 5. A four-cold-process,
no-sibling AUTO spike reproduced the baseline route instability: three full
512-token outputs were byte-identical and the fourth forked at byte 1,088.
The production receipts explain why timing matters: one OFF trial-2 process
cost-disabled both proposers and issued 408 ordinary target forwards, while
its replica kept MTP/history speculation live and issued zero ordinary
forwards. Exact decoding must remain route-equivalent despite that adaptive
choice; the failing gate is therefore retained, not relaxed.

The deciding fixed-route spike forced speculation OFF while preserving one
live 256-token decoder plus four simultaneous cold prefills, two fresh OFF
processes and two fresh ON processes. All four decoder responses were
byte-identical (content SHA-256
`e54a3f5221d9da568a40d1eb37566665cf472ff1f2931f2c05b5576ed70ff14a`),
all sixteen prefill responses were exact cold `OK`, OFF published no
rectangular cohort, and each ON process published one real width-four cohort.
Evidence is under
`/opt/hf2q-evidence/qwen38-mixed-fixed-route-25f498c5`. This proves the
rectangular transaction is coherent at the reproduced fork and reformulates
the blocker as AUTO target-route equivalence: production scalar
`forward_gpu_greedy` versus the four-row full-logit verifier at the observed
long prefix.

Rev 45 closes that deciding spike without weakening the gate. First, the
exact 512-token request was run with speculation OFF through both ordinary
decision surfaces: the fused scalar greedy route and the full-logit CPU
sampler route produced the same 2,534-byte content, SHA-256
`cf783007305a9a02c4b56e5c5d511c2aa6d0b28977d01a3b607fd02478ae04f8`.
That rejects ordinary argmax/readback as the fork.

The real-artifact repeated-verifier test then reproduced the production
shape from source rather than a synthetic prefix. It rendered and tokenized
the exact request into 67 tokens, proved the stable boundary is the strict
60-token prefix, ran independent byte-identical `60 + 7` compound target/MTP
prefills, and compared the production scalar target trajectory against 127
successive full-accept four-row target/MTP transactions. The target routes
diverged at verifier round 51, post-seed decision offset 205 (completion token
206), row 1, with inputs `[381, 17083, 440, 16866]`: scalar selected token
`440`, while the four-row verifier selected `18546`. The verifier logits were
near a real decision boundary: top one `(18546, 22.945137)`, top two
`(440, 22.900723)`, margin `0.04441452`. The first hardware invocation carried
an extra non-production `MLX_UNRETAINED_REFS=1` and failed pre-comparison with
Metal invalid-resource; the valid production-environment rerun omitted it and
reproduced the same round-51 fork twice, including after byte-proving the two
compound prefill snapshots. The checked-in test is
`qwen38_real_repeated_four_row_route_coherence`; the artifact is the exact
Qwen3.8 Q5_K_M file with SHA-256
`4b19f41c391d962882e459be3315d4e3c54079892db2848f66b78815b185156e`.

Rev 46 identifies and falsifies the route defect. Source inspection found that
the Qwen3.8 loader changed the shared native routing policy from byte-exact
Q6_K multi-row matvec/default-off `mul_mv_ext` to `dense_decode_mvn=false` and
`dense_decode_mv_ext=true`. Scalar target decode therefore used the ordinary
Q5_K/Q6_K matvec reduction, while the four-row verifier used `mul_mv_ext`,
whose implementation and earlier ADR already document a different reduction
tree. The earlier four-position gate was too short to expose accumulated drift.

The deciding real-artifact rerun preserved the same 67-token prompt, 127
four-row verifier rounds, TQ cache, compound target/MTP prefill, and every
target comparison; it changed only the two routing overrides to
`HF2Q_DECODE_MV_EXT=0` and `HF2Q_DECODE_MVN=1`. All 508 post-seed decisions
were then exact in 39.16 seconds. This exonerates scheduler order, rectangular
prefill, history lookup, ordinary argmax, compound prefill, full-attention
routing, and recurrent DeltaNet for this incident. The Qwen3.8-local routing
exception is removed. Qwen returns to the shared coherent native defaults:
Q6_K uses the already byte-proven multi-row kernel and Q5_K temporarily uses
the row-independent ordinary matvec at width four.

The performance follow-through is universal rather than model-labelled: add a
Q5_K multi-row kernel that literally preserves the scalar Q5_K accumulator and
reduction order while amortizing each weight read across columns, then admit it
through the shared `dense_decode_mvn` capability for every applicable family.
A guessed logit-margin epsilon and the non-bit-exact `mul_mv_ext` route are not
accepted fixes. After the shared kernel lands, AUTO must re-pass same-policy
identity before OFF/ON identity and speed acceptance. Qwen3.6 Mixed remains
unrun because the joined matrix stopped at the first coherence failure.

The corrected hf2q loader and every production Qwen matrix now use the shared
policy rather than the artifact-name exception. The permanent ignored gate
asserts both routing variables are absent before load and asserts the resolved
model policy is exactly `dense_decode_mvn=true` and
`dense_decode_mv_ext=false`. On the same Q5_K_M artifact and 67-token prompt,
that production-default gate passed all 127 rounds / 508 post-seed decisions in
38.03 seconds. Shell syntax, the Qwen3.8 artifact-matrix contract, the
cross-family MTP artifact contract, and the shipping-contract suite also pass.
This accepts the coherent route correction; it does not yet accept the open
Q5_K speed replacement or restore universal AUTO performance authority.

Rev 48 closes the remaining DenseQ function-identity gap found during that
route audit. Scalar/pooled DenseQ used a fused gate/up/SILU entrypoint for
native IQ4_NL, Q4_K, Q5_K, Q6_K, and Q8_0 weights, while the arena-backed
multi-row verifier always used separate gate, up, and SILU dispatches. The
native tolerance tests cannot establish greedy-trajectory equivalence here:
the Q5_K fused kernel's different FMA grouping produces small nonzero changes.
A focused hf2q Metal regression reproduced the defect before implementation
at width four (`-3.235553158e-4` versus `-3.235536860e-4` in the first output
element). Eligibility and codec dispatch now live in one helper used by both
pooled and arena-backed execution; the down-projection barrier and projection
route are unchanged. The strengthened regression compares every output bit for
each row at widths two through eight across all five eligible codecs and passes
35/35 codec-width cells. This is exact synthetic operator evidence, not a
full-artifact speed claim. The exact Qwen3.8 Q5_K_M repeated-verifier/AUTO gate
and matched OFF/ON performance measurement remain required on the landing
commit; a prior rejected custom cross-row fusion spike is not evidence against
this row-independent production kernel reuse.

Rev 49 makes the pending exact-Q5 proof non-vacuous before changing the pinned
backend. The full-model artifact contract now binds each qualified width-four
format to its required native multi-row kernel label: Q4_K_M to Q4_K, Q5_K_M
to Q5_K, and Q6_K to Q6_K; BF16 and Q8_0 correctly require no K-family label.
The repeated-verifier gate additionally requires the served artifact to report
GGUF file type 17 (Q5_K_M), resets dispatch telemetry immediately before the
127-round comparison, and accepts only an observed Q5_K `r1=4` route after all
508 scalar-versus-four-row decisions agree. The model-free contract and two
mutations (missing Q4 evidence and a substituted Q5 route) pass.

The deciding pre-pin falsifier used the then-current published `mlx-native 0.13.0`,
the exact Qwen3.8 Q5_K_M artifact and production-default route with
`MLX_DISP_BUCKET=1`. It deliberately failed only at the new final canary after
42.50 seconds: all 508 decisions agreed, scalar Q5_K executed 193,294 times,
Q6_K `r1=4` executed 8,382 times (proving telemetry was live), and Q5_K
`r1=4` executed zero times. Therefore the gate is capable of distinguishing
the old row-independent fallback from the exact shared-weight Q5_K kernel; a
passing result is reserved for the published, checksum-pinned backend that
actually dispatches that kernel. No speed or release claim is made by this
falsifier.

Rev 50 closes that coherence and dependency-publication hypothesis without
claiming the still-unmeasured application speedup. `mlx-native 0.14.0` was
published from merged main commit
`32f076c7502151e7ca9cb20c06d0f3fe5e1d5641`; protected workflow run
`32873363483` passed exact-source and packed-crate all-feature tests, crates.io
publication and byte verification, a downloaded-registry test, tag/release
creation, and GitHub release byte verification. Independent downloads of both
registry and GitHub assets produced the exact crate SHA-256
`c7b359aa9ea2603f58b49151ba54e37ed1aac10e76faf530865ea30a95f051b4`, and
tag `v0.14.0` resolves to that exact commit. hf2q now has one exact crates.io
lock record for that version and checksum; full locked metadata, dependency
provenance, shipping, artifact-matrix, and generative-swap contracts pass.

The same 67-token Qwen3.8 Q5_K_M repeated-verifier gate then passed all 127
rounds / 508 post-seed decisions in 42.81 seconds and emitted the observed
`kernel_mul_mv_q5_K_f32_mN_r1_4` route canary. The stored-execution evidence
fixture exposed and corrected one stale expectation from rev 48: prompt and
decode now each record 14 operations because both layers use the shared fused
gate/up/SILU identity, for 28 total rather than the former 30-operation mix of
separate prompt and fused decode paths. The exact binding set, catalog seal,
and substituted-backend-version mutation pass. The historical ViT arithmetic
pin is unchanged. These are dependency, route, and trajectory receipts; the
clean release-binary OFF/ON ABBA remains the speed authority.

Rev 51 rejects the assumption that a single-dispatch fused Q5_K gate/up/SILU
kernel is faster than the exact separate route merely because it encodes fewer
GPU dispatches. On the exact `54898f06c9f9d3ee1c345cb4ed02aa34fd5faed7`
hf2q source and binary SHA-256
`4d99f0c3c1f9f6de748ebabbb632f28a4dd6f03047ddca2699cb75c3dc47aa85`,
the five-width physical gate ran the hash-bound 19,535,701,568-byte Qwen3.8
Q5_K_M artifact with speculation off and exact scalar replay per lane. The
unfused receipt is
`/opt/hf2q-evidence/qwen38-q5-fusion-off-54898f06/summary.json`; the immediate
fused confirmation receipt is
`/opt/hf2q-evidence/qwen38-q5-fusion-on-recheck-54898f06/summary.json`. Both
passed all widths. Completed-wave throughput for unfused versus fused was
22.582/11.637, 30.845/14.491, 34.158/17.388, 29.903/16.574, and
24.161/16.955 tokens/s at widths 1/2/4/8/16: respective speedups of 1.940x,
2.129x, 1.964x, 1.804x, and 1.425x. Command-buffer submissions were unchanged
at every width, falsifying submission count as the explanation. Worst TTFT
also improved at every width.

The resulting product decision is smaller than a new kernel: Q5_K is removed
from hf2q fused-FFN eligibility and uses the already exact separate gate/up
route for scalar and rectangular execution. hf2q's unused Q5 fused wrapper and
trace surface are removed; qualified IQ4_NL, Q4_K, Q6_K, and Q8_0 fused routes
remain unchanged. A future weight-sharing fused Q5 kernel is an open
hypothesis, not a dependency: it may replace the separate route only after
bit-exact scalar authority and a matched physical ABBA beat this measured
default. The production-default clean-binary rerun remains the landing proof.

The first BF16 matched-peer ABBA at the same source commit is retained only as
falsification evidence. hf2q's median internal decode rate was 28.411 tokens/s
versus 27.333 for the pinned peer, but the calibrated code and repeat speed
bands overlapped; the gate correctly returned `inconclusive, not parity` and
published no summary. Independent review also found that its sealed launch
receipt inverted the actual MVN/MV_EXT settings, and that related matched-
physical contracts used a non-causal speculation window, engine-specific
token counters as a cross-engine code unit, and ambiguous per-engine cache and
speculation labels. Those proof contracts must derive and validate actual
launch policy, bracket only timed waves, use common task-wall/request units,
and bind each engine's cache/speculation configuration before any matched-peer
speed result can be accepted.

Rev 52 records the clean production-default proof after the Q5 decision
landed. Runtime commit `e260fade641933d7453ddb5b42b7214f9a9bd725` combines
the exact Q5 route correction with causal matched-reference receipt
contracts. Its
release binary SHA-256 is
`770e9e7f050459d0922fe2766684127fe612b8807a293d0c4387d86f4156626a`.
With no `HF2Q_FUSED_GATE_UP_SILU` override, receipt
`/opt/hf2q-evidence/qwen38-q5-production-e260fade/summary.json` passed exact
scalar replay at physical widths 1, 2, 4, 8, and 16. Completed-wave throughput
was 22.125, 29.716, 33.781, 34.110, and 41.964 tokens/s. Relative to the
immediate fused confirmation in rev 51, those results are 1.901x, 2.051x,
1.943x, 2.058x, and 2.475x faster; internal aggregate decode throughput also
improved at all five widths. Model-free policy, five-codec exact-operator,
four-position real-artifact, and repeated 508-decision gates are green. The
real-artifact gates require the native Q5 multi-row route canary and reject the
retired fused-Q5 route canary. This proves the production default and its
format/width coherence; it does not substitute for the still-open causal
matched-reference and full agentic-lifecycle authorities.

Rev 53 records the deciding causal comparison and reformulates the next
hypothesis from measured phase data. Exact commit
`634d638c1c5ffd3b7d5183e905ff7a7411c9156f`, binary SHA-256
`e6b5900860a12b9f084ff6af1a2f9c51516aa489238e229a7949df6aeca00880`,
and the rev-52 Q5 artifact completed a stable four-leg hf2q/reference/
reference/hf2q run at
`/opt/hf2q-evidence/qwen38-q5-matched-peer-634d638c`. Quality and thermal
checks passed, but the speed gate correctly refused a summary: hf2q's worst
code/repetition end-to-end rates were 30.675/26.474 tokens/s versus the
reference's best 38.877/33.941. Speculation acceptance was not the cause:
hf2q accepted about 95.2% of measured draft tokens versus about 94.5%, with
161 versus 165 target-verification rounds.

The follow-up phase receipt
`/opt/hf2q-evidence/qwen38-q5-mtp-phase-634d638c-v2/server.log` captured 36
exact K=3 rounds. Mean target verification was 90.99 ms. The instrumented
GPU-busy bucket reported 5.49 ms, but this is only a lower bound because it
records blocking submissions and excludes asynchronous GPU execution; the
remaining 85.51 ms is therefore unattributed residual, not CPU time. Every four-row verifier emitted
exactly 163 command buffers/submissions, 18 host synchronizations, and 1,347
dispatches. Source inspection then found a narrower live predicate than the
documented GPU-only contract: resumed short full-attention ops1-4 treats
`gpu_only_prefill_resume` as non-blocking, but ops6-7 ignores the same proven
condition and executes a blocking wait once per full-attention layer. The
first deciding spike therefore adds that existing predicate to the ops6-7
non-blocking branch. It is accepted only if four-row/scalar trajectory and
cache handoff remain exact, synchronization count falls from 18 to the
structural terminal floor, and normal-mode verifier wall improves. If the
remaining 163 submissions still dominate, the next measured hypothesis is
to compose the existing TQ write, direct packed-KV attention, and ops6-7
dispatches into the caller-owned layer encoder without changing arithmetic.

Rev 54 closes that first hypothesis as falsified. Exact spike commit
`c6172f8f98ea93199201f29e7ce0484a904ad2fc` and release-binary SHA-256
`4e6862fd8c650aa1beec842832c52a35d08ffec9f67f22591b4d2a503127293c`
passed Q5 four-row/scalar trajectory, native-route, and cache-continuation
proof. Its matching 36-round receipt at
`/opt/hf2q-evidence/qwen38-q5-mtp-phase-c6172f8f/server.log` reduced every
verifier from 18 synchronizations to 2 while preserving 163 command buffers,
163 submissions, and 1,347 dispatches. Mean verifier wall moved only from
90.99 ms to 90.34 ms (0.7%, below the noise/acceptance floor), and output
content was byte-identical. The wait-only code change is therefore removed
from the landing diff. The deciding next hypothesis is submission
composition, specifically a caller-encoded resumed TQ full-attention stage;
its falsifier must reduce actual submissions, preserve all dispatches and
exact trajectory/cache state, and beat the 90.99 ms baseline in normal mode.

Rev 55 closes submission composition as falsified. Exact spike commit
`ebc1105205a6881b0ebfbbc6b07cdf03a14d4428` and release-binary SHA-256
`8670a60826453f0ac5490635dfeb8c197555173a5ceebc8b15e46e14ecea0707`
passed the native Q5 four-position, segmented-boundary, rectangular-state,
cache-continuation, and caller-encoded direct-TQ equivalence gates. Its
36-round receipt at
`/opt/hf2q-evidence/qwen38-q5-mtp-phase-ebc11052/server.log` reduced every
verifier from 163 to 115 command buffers/submissions and from 18 to 2 host
synchronizations while preserving all 1,347 dispatches. Generated content
remained byte-identical (SHA-256
`8bfc5303550444ed92150d1375a07d2f2d9d57ab64a3a86ac401630ed86acd2b`).
Mean verifier wall improved only from 90.99 ms to 89.45 ms (1.7%), and request
decode rate from 37.10 to 37.90 tokens/s (2.2%); both miss the pre-registered
5% acceptance floor. The composition code is removed from the landing diff.
The next deciding measurement must attribute asynchronous GPU execution by
kernel on the unmodified production graph. A following optimization is
admitted only when it targets measured dominant kernel work or increases
useful accepted tokens per target verification; submission-count reduction
alone is no longer a sufficient hypothesis.

Rev 56 corrects the first asynchronous GPU attribution before admitting the
next spike. The clean Metal trace is
`/opt/hf2q-evidence/qwen38-q5-metal-trace-8d707c3a/metal-clean.trace`;
its exported encoder and GPU-interval tables contain no inherited process
environment. A direct grouping by command-buffer label assigned 2,083.922 ms
to `layer.gdn.stage_a` and 700.013 ms to `layer.full_attn.ops1-4`, but those
names are termination boundaries rather than exclusive operator ranges when
`HF2Q_ENCODER_SESSION=1`. Every nonterminal DenseQ FFN ends with
`carry_into_next_stage`, whose sessioned arm emits only a memory barrier. The
next attention fence therefore commits the still-open command buffer and owns
the trace label for both the preceding layer's residual/FFN work and current
attention Stage A. The trace independently fits that source fact: interval
counts are 1,946:647 (3.008x, the model's 48:16 linear/full-attention layer
ratio) while mean interval GPU times are nearly equal at 1.0709 and 1.0819 ms.
The trace does **not** prove DeltaNet Stage A consumes 52% of verification;
that interpretation is rejected. A future phase receipt must either terminate
at exclusive operator boundaries or enable per-kernel shader attribution.

The next admitted hypothesis is narrower and independently measured. On clean
`mlx-native` main `32f076c`, the checked-in single-tenant Q5_K width-four test
at `M=4, N=K=5120` measured scalar-tree 98.486 us, current exact mN 79.683 us,
and the inexact reduction route 46.929 us (medians of five 100-call GPU-timed
samples). The faster route is not admissible because it changes the reduction
tree and already failed whole-model trajectory equality. H3 instead pairs two
output rows per SIMD group in the exact mN kernel and reuses each activation
tile across the pair, while retaining each row/column's scalar block walk,
accumulation order, and `simd_sum` tree. This is the Q5 analogue of the existing
exact Q6 NR2+mN structure and is codec-wide rather than Qwen-specific.

H3's smallest spike uses the served artifact's real Q5 shapes (`K=5120`,
`N=10240` QKV and `N=6144` gate), plus odd/tail shapes and physical widths
2 through 8. It must be bit-identical to the scalar authority for every output
and improve the median GPU time of both real shapes by at least 15% over the
current exact mN kernel before production routing changes. A promoted candidate
must then preserve the Qwen3.8 four-position and repeated-verifier trajectory,
cache continuation, all-format/physical-width matrix, and improve normal-mode
mean verifier wall by at least 5% against the 90.99 ms rev-53 baseline. Missing
either speed floor falsifies H3: remove the routing/kernel candidate, retain
only the measured conclusion, and reformulate from exclusive kernel evidence.

Rev 57 closes H3 and its two narrower reformulations as falsified. The first
paired-row implementation reused each activation tile across two output rows.
The existing width-2-through-8 byte gate stopped it before performance: at
width 2, row 1 changed from `5.674408` to `5.674407` (two F32 ULPs). The
activation-sharing form is therefore incoherent and removed. A geometry-only
pair then ran the original mN body independently for each row. It passed exact
parity at every production width, but improved the 5120x5120, 6144x5120, and
10240x5120 kernels by only 0.9%, 1.0%, and 1.2%, respectively; it misses the
15% kernel floor and is removed.

The second reformulation explicitly loaded each block's 24 packed Q5 bytes
once before the four-column loop, without moving a floating-point operation.
It remained bit-identical but regressed all five measured shapes: current
exact mN versus packed-hoist GPU medians were 79.689/80.591 us (5120x5120),
95.671/96.785 us (6144x5120), 159.621/161.514 us (10240x5120),
272.662/275.264 us (17408x5120), and 285.957/287.441 us
(5120x17408). The compiler's existing packed-load plan is better; that code
is removed. Finally, a four-SIMD-group scheduling spike exceeded this heavy
kernel's executable threadgroup shape and left guarded output rows unwritten,
so it did not reach a timing gate. The original two-group geometry is restored.

No Q5 production kernel or route changes survive rev 57. The useful benchmark
extension retains the real QKV, gate, FFN gate/up, and FFN down shapes. The
next optimization may not infer a hot kernel from compound command-buffer
labels: it must first obtain exclusive per-kernel shader timing or add
operator-exclusive measurement boundaries, then pre-register a lever against
the measured winner. The 46.9-us inexact route remains a headroom indicator,
not an admissible implementation.

Rev 58 admits the next measurement, not an optimization. A diagnostic-only
`HF2Q_QWEN_EXCLUSIVE_PROFILE=1` spike must fail closed unless encoder-session
composition is disabled, then use mlx-native's existing
`CommandEncoder::profile_stage_boundary` to terminate and GPU-time Qwen
verifier operator groups. It separates dense FFN gate+up, SiLU, and down;
DeltaNet pre-norm, QKV projection, gate projection, convolution/capture, and
split; and the recurrent branch's norm/projection/recurrent/output groups.
With the preceding FFN no longer carried into the next layer's command buffer,
the existing full-attention stage labels also become attributable apart from
their small residual-add predecessor. Profile-mode request wall is not a speed
claim: synchronous boundaries deliberately destroy production overlap. The
deciding receipt is the GPU-time ordering and reachable share of those groups,
cross-checked against the clean production trace and model-free operator
microbenchmarks. The spike selects exactly one next performance hypothesis;
its instrumentation is removed from the landing diff unless a focused test
proves the opt-in diagnostic itself remains useful and default-off.

Rev 59 closes that measurement and rejects both attention and submission
overhead as the next primary target. The diagnostic source was based on exact
commit `cabbf44e8c169c42031daefb5b00a239cb9279f4`; its release binary SHA-256
was `444e258c4b06406d31798118672cba31ad83abe5bcc935621d8c19643c606cab`.
It served the same 19,535,701,568-byte Qwen3.8 Q5_K_M artifact, prompt,
temperature-zero settings, single physical slot, TQ KV, and AUTO speculation
as revs 52-55. The 256-token request reproduced the exact accepted Rust
completion from the production trace. A second 32-token request reused 59
cached prompt tokens and supplied eight width-four verifier forwards for the
deciding aggregate.

Those eight forwards measured a 102.625 ms median total under the deliberately
serialized diagnostic boundaries. Dense FFN gate+up was 42.640 ms median and
down was 20.590 ms: 61.6% of measured GPU time together. The next largest
groups were DeltaNet QKV projection at 8.555 ms, the post-recurrent/out
projection range at 6.700 ms, DeltaNet gate projection at 6.610 ms, and the
MTP shared head/argmax at 5.740 ms. Full-attention ops1-4 and ops6-7 together
were 7.100 ms (6.9%). A width-seven verifier sample independently assigned
66.5% to the same dense FFN projections. The synchronous profile wall is not
compared to the 90.99 ms production wall; only operator ordering and reachable
share are authoritative.

The diagnostic boundaries are removed from the landing diff. The next
hypothesis must target the exact native Q5 dense FFN projection route while
preserving weight bytes, per-output arithmetic/reduction order, and target
trajectory. It may not repeat the already falsified fused-Q5, shared-
activation NR2, geometry-only row pairing, packed-byte hoist, or four-
SIMD-group scheduling spikes. Its smallest gate is model-free byte equality
at the served 5120x17408 gate/up and 17408x5120 down shapes, followed by a
matched real-artifact verifier test. Production admission still requires at
least 5% lower normal-mode verifier wall with identical output/cache state;
otherwise the candidate is removed and the measured conclusion retained.

Rev 60 admits the first source-bounded reformulation from that profile. The
exact Q5 mN kernel currently executes two independent SIMD groups per
threadgroup, one output row per group. A four-group expansion already failed
the executable heavy-kernel shape in rev 57, but the opposite occupancy
hypothesis remains untested: one SIMD group per threadgroup may permit more
resident groups or avoid threadgroup-level register pressure while leaving the
packed-weight walk and every floating-point operation unchanged. H5 therefore
duplicates only the dispatch geometry and row mapping, not the arithmetic
body. It must first match scalar authority bit-for-bit for physical widths
2 through 8, including odd/tail rows. It proceeds only if the weighted median
GPU time across the served FFN gate/up (`5120x17408`) and down
(`17408x5120`) shapes improves by at least 10% over the current exact mN
route. A miss removes the candidate and records the result; a pass advances
to the rev-59 whole-model 5% verifier gate.

H5 was bit-identical to scalar authority at every physical width, but failed
the performance gate and is removed. Current two-group versus one-group GPU
medians were 79.745/80.213 us (`5120x5120`), 95.759/96.179 us
(`5120x6144`), 160.956/161.643 us (`5120x10240`), 274.953/278.031 us
(`5120x17408`), and 288.274/289.209 us (`17408x5120`). Every shape
regressed by 0.3-1.1%; the gate/up/up/down FFN-weighted result regressed 0.85%.
The production kernel, dispatcher, and dispatch-trace contract were restored
exactly. Only the real-shape benchmark extension remains as reusable evidence.

Rev 61 admits H6 from a fresh source audit. In the exact Q5 mN shader, `R1`
is compile-time but `nb = K / 256` remains runtime. The served gate/up shape
always has `K=5120`, hence exactly 20 blocks and five `i += 4` iterations per
lane. H6 adds one function constant for that block count and binds it only on
the exact Q5 `K=5120` route; every other K retains the generic runtime value.
It changes no buffer layout, dispatch geometry, weight bytes, per-column
loads, floating-point operation, or reduction order, and creates no converted
or repacked model state. The parity gate covers odd N, poisoned output, and
physical widths 2 through 8 against scalar `to_bits`, plus generic fallback
at other K. Performance uses paired warmed medians on the real
`5120x17408` gate/up shape. A stable projection win of at least 2% advances
to the exact whole-model verifier gate; landing still requires a reproducible
positive normal-mode verifier delta with identical output/cache state. A miss
removes the function constant and records the conclusion.

H6 remained bit-identical at every specialized physical width, but missed its
speed floor and is removed. Candidate commit `502b3d0` and restored-control
commit `84863cb` used the same extended real-shape benchmark. Across three
warmed runs per state, the real `5120x17408` gate/up GPU-time medians were
269.903/269.772/269.751 us specialized versus
274.949/274.821/274.765 us restored, a 1.84% median time reduction rather
than the required 2%. The other K=5120 shapes reduced median time by
1.5%, 1.7%, and 2.8%; the unspecialized K=17408 down projection was
unchanged within noise. Because gate/up is 41.5% of the measured verifier,
the observed candidate reaches only about 0.8% of verifier wall before other
overheads. No whole-model claim is made. FC703, its route binding, and its
dispatch-trace surface are absent from the restored source; the reusable
real-shape benchmark remains.

Rev 62 admits H7, an exact input-traffic hypothesis with enough measured reach
to clear the product floor. The current Q5 mN threadgroup runs two independent
SIMD groups for two output rows; both groups load the same F32 activation
values while walking different packed-weight rows. H7 leaves those row-local
weight, scale, accumulator, and `simd_sum` operations untouched, but has the
first group stage each activation tile once in threadgroup memory for both
groups. This is not the rejected NR2 sharing form (one group never owns two
row accumulators), not a cross-projection/fused FFN kernel, and not a weight
repack. The kernel retains two groups and native GGUF views. Uniform barriers
require the packed block count to be divisible by four; other shapes retain
the current kernel. Odd output rows must remain bounds-safe without allowing
one group to skip a threadgroup barrier.

The smallest gate compares scalar `to_bits` at physical widths 2 through 8,
odd N, poisoned output, and nonzero logical views, then GPU-times the served
`5120x17408` gate/up and `17408x5120` down shapes. H7 proceeds only if the
gate/up/up/down weighted median improves by at least 10%. A passing primitive
then requires the Qwen four-position, repeated-verifier, cache-continuation,
all-format/width, and normal-mode whole-model gates, including at least 5%
lower verifier wall with byte-identical generated content and cache state.
The primitive is codec-generic across dense Q5 projections; family graphs do
not get separate implementations.

H7 passed scalar byte parity at every production width, including the odd-row
barrier case, but regressed far before the 10% speed gate and is removed.
Candidate commit `933df9a` measured 519.427 us on the real gate/up shape and
524.277 us on down, versus the restored controls' approximately 274.8 and
288.3 us. The gate/up/up/down weighted projection cost rose about 86.5%.
Ten uniform barriers per K=5120 walk plus 16 KiB of threadgroup storage at
width four cost much more than the duplicate activation reads; cache already
serves those reads cheaply. Restore commit `a6ebf5a` removes the staged kernel,
registry names, shared-memory dispatch, and trace contract exactly. The result
also rejects the more complex cross-projection staged variant: it would pay
the same storage/barrier cost while additionally replacing two already-
concurrent dispatches.

Rev 63 admits H8, a deliberate arithmetic-authority spike rather than another
attempt to preserve the slower scalar tree. The existing Q5 q4x4 route is
about 41% faster at width four, but it cannot be used only by the verifier:
ordinary one-row target decode would then follow a different reduction tree
and exact speculative verification would be impossible. H8 therefore adds a
Q5-only q4x4 `R1=1` kernel and fixes `nxpsg=8` for Q5 at every physical width
1 through 8. Every column of R1=2/3/4/5 must match an independent R1=1
dispatch bit-for-bit. Q4 and Q6 retain their current authorities. Native Q5
GGUF bytes are read directly and dequantized inside the kernel; no stored
dequantization, requantization, interleaving, or model-state rewrite is
permitted.

The gate sequence is fail-closed. First, model-free tests cover distinct Q5
bytes and inputs, odd N, poisoned outputs, logical views, K=5120 plus a second
valid K, tail widths, and runtime/precompiled pipelines. Second, paired GPU
timing covers both M=1 and M=4 at the served gate/up and down shapes; a faster
verifier path is rejected if ordinary M=1 erases the end-to-end gain. Only
then may a codec-specific frozen routing policy precede current Q5 mN. The
candidate policy must pass 127-round one-row/four-row decision and cache-state
equality, speculation OFF/AUTO exact target trajectory, fixed-corpus quality,
agentic tool semantics, and at least 5% lower normal verifier wall. Equality
to the former scalar-tree trajectory is not claimed; former-versus-candidate
is a quality comparison. The policy is codec-wide for native dense/shared Q5
projections across applicable families, while routed expert kernels remain
separate.

Rev 64 accepts H8's deciding spike and promotes it to a release candidate.
The initial mlx-native candidate `e87a63d` passed model-free `to_bits`
equality for packaged and runtime-compiled Metal, widths one through eight,
odd output rows, distinct inputs, nonzero logical views, and buffer canaries.
The specialized implementation remains internal to the validated canonical
matmul dispatcher; model weights stay in their native Q5_K GGUF blocks. The
codec-wide default is enabled independently of model labels, with an explicit
diagnostic opt-out. The obsolete fused Q5 gate/up operator is removed because
its width-invariance did not make it numerically coherent with this canonical
projection authority; hf2q had already stopped routing Q5 through it.

On the exact 19,535,701,568-byte Qwen3.8 artifact (SHA-256
`4b19f41c391d962882e459be3315d4e3c54079892db2848f66b78815b185156e`),
127 verifier rounds produced 508 identical target decisions and cache state.
A paired same-process verifier measurement reduced total wall from 6,105.525
ms to 4,527.208 ms (25.851%), with round medians of 94.818 ms and 70.389 ms.
The fixed 139-sample corpus retained 1.0 argmax agreement; perplexity moved
from 1.514567 to 1.513641. The server OFF/AUTO gate retained all 24 exact
target choices while the two accepted workloads improved by 83.36% and
127.54%. A four-slot agentic lifecycle additionally passed tool-call and
tool-result continuation, cancellation, queued exact retry, prefix-cache
reuse, isolation, and execution-identity checks.

The separate 25,043,007,488-byte Qwen3.6 Q5_K_M artifact proved the same
canonical route at one and four rows while its 120 observed expert calls
remained on `mm_id`; this closes the dense/shared applicability question
without rerouting MoE experts. The staged Gemma4 and DeepSeek4 artifacts
contain no Q5_K tensors, so their applicable proof is the codec-level
model-free battery plus unchanged family gates, not a fabricated Q5 execution
claim. The fresh-process ABBA timing was rejected because order/thermal spread
dominated the candidate delta; only the paired same-process verifier result is
admitted above.

Rev 65 publishes that backend from merged mlx-native main commit
`25b4eab6cd2de0d790c34d046473febf798543c4`. Protected release run
`32917226470` passed immutable identity, minimum Rust, semver, audit, exact-
source and packed-archive all-feature tests, crates.io publication and byte
verification, downloaded-registry tests, tag/release creation, and GitHub
release byte verification. Independent crates.io and GitHub downloads both
hash to `09d3decffbf66811bac728abd51697c89cd699e031bc1b4295470108f235b822`,
and tag `v0.15.0` resolves to the merged commit. hf2q now requires exactly
`mlx-native = "=0.15.0"` and its sole lock record names the registry source and
that checksum; the temporary local Cargo patch is absent.

The first registry-pin falsifier also caught two integration defects before
hardware qualification. The generic F32-keep predicate preempted Apex's
explicit Q5_0 router policy, so an end-to-end converted router reopened as F32;
planner precedence now preserves Apex Q5_0 while leaving standard conversion
at F32, and the complete 18-test conversion integration suite passes (17 pass,
one pre-existing ignored RSS fixture). A GPU lock-discipline test initially
stopped at a missing acquisition, then exposed an existing acquisition that
was not the first statement; both proof defects are fixed. The complete binary
suite now passes 5,121 tests with 66 explicit ignores and zero failures.

This is not the final hf2q release claim. Clean-source all-format four-
position and physical multi-slot gates are now complete; model-swap, full
agentic lifecycle, and matched-reference gates remain required. Those
harnesses must reopen the actual frozen process-policy log; an environment
value or receipt field alone is not execution proof.

Rev 66 seals the clean-source Qwen3.8 artifact matrix at runtime commit
`6fad526d687b7a8d4b9e9a1c95468468ee3c661e`. BF16, Q4_K_M, Q5_K_M, Q6_K,
and Q8_0 each passed exact four-position/state continuation and real physical
widths 1/2/4/8/16 with per-lane scalar replay. The Q5 receipt reopens the
actual `kernel_mul_mv_ext_q5_K_f32_r1_4` dispatch canary. The immutable
receipts are `/opt/hf2q-evidence/universal-release-6fad526d-four-position/matrix.json`
and `/opt/hf2q-evidence/universal-release-6fad526d-physical-multislot/matrix.json`.

The first cross-family swap spike then failed before model admission for a
proof-integrity reason: the supplied sealed binary lived in the source Cargo
target, and the gate's own `cargo test --release` replaced that pathname
before the Rust harness checked its SHA-256. The observed digest mismatch was
therefore a harness/candidate identity race, not a model-load result. The
reformulated contract snapshots the already-validated candidate into a private
temporary executable before hashing or harness compilation. The same path is
then executed and re-hashed after the run, so Cargo cannot replace the bytes
being attested. The cross-family real-artifact spike must be rerun from a new
exact commit before its acceptance box can close.

The corrected runner then reached the first real Qwen-dense admission and
rejected 18,275,182,400 resident bytes against the qualified 16,810,714,944-
byte text artifact. Source inspection showed that the selected local directory
also contained the automatically paired projector: pool accounting correctly
charged text bytes plus page-rounded projector mappings plus its complete
vision-cache reservation. This was not an accounting regression. ADR-047's
universal generative chain is the text-only gate; its distinct multimodal
projector lifecycle has a separate acceptance row. The reformulated runner
therefore requires each qualified text GGUF in an isolated artifact directory
and rejects projector/other GGUF siblings before loading. The real rerun uses
same-inode hard links in isolated directories, preserving the exact qualified
text bytes without a copy or format rewrite.

Rev 67 records the next swap falsifier without relaxing the product clock.
The corrected text-only candidate passed its initial Qwen phase, but its first
DeepSeek activation took 202.787 seconds. A live process sample placed the
delay in `TextArtifactIdentity::inspect` after the Qwen engine had shut down;
the actual DeepSeek map plus synchronous warmup was about 14.8 seconds. The
universal runner, unlike the already-sealed exact-Qwen swap runner, had not
prepared or passed the server's canonical schema-v2 verification-receipt
directory. The server therefore performed the correct but availability-hostile
107 GB content hash inside the replacement transaction.

The reformulated runner hashes or cache-validates all four stable artifacts
through the sealed binary before server startup, passes the closed receipt
directory to the existing canonical-path identity registry, and seals every
receipt in the evidence manifest. The contract rejects any missing, extra,
legacy, wrong-path, wrong-digest, or stale-snapshot member; its explicit stale
snapshot mutation is green. Transition time remains the complete
client-observed activation under the fixed 60-second ceiling; no timing
exclusion or new runtime hash path was added. This is source/model-free
authority, not a real-artifact pass; the fresh 13-phase rerun remains required.

The first exact-candidate invocation stopped before hashing or model admission
because the runner inherited an operator Cargo configuration forbidden by its
own exact-source policy. The runner now uses one absolute isolated Cargo home
outside source and evidence trees for every build, metadata, and test command;
the no-ambient-patch authority remains unchanged.

The next exact run passed the formerly slow preverified DeepSeek transition
and reached Qwen-MoE, then the host-wired gate rejected a 115,267,158,016-byte
peak. Diagnostic replay showed that peak was below the 115,271,286,784-byte
pre-switch DeepSeek endpoint; the old 34,967,713,792-byte bound had admitted
only the smaller destination and therefore rejected the source at sampler
start. The source/destination-symmetric contract now admits the maximum of
either endpoint plus margin or the lower endpoint plus one destination and
margin. It still rejects simultaneous DeepSeek plus Qwen-MoE residency. The
source-dominant mutant and independent shell arithmetic are green; a fresh
exact-chain rerun remains required.

The fresh run then passed Qwen-Dense, Qwen-MoE, and DeepSeek before the Gemma
phase exposed a proof-classification bug rather than a runtime ownership leak:
`lsof` found no live Qwen-MoE descriptor, but the `vmmap` fallback searched
`APEX-Q5_K_M.gguf` as a basename substring and therefore matched Gemma's
distinct `gemma4-ara-2pass-APEX-Q5_K_M.gguf` path. The classifier now requires
the exact canonical artifact path as a complete trailing `vmmap -wide` field;
the focused collision mutant is green and the fail-closed eviction assertion
is unchanged. The complete exact-chain rerun remains the hardware authority.

That corrected runtime completed all thirteen phases and twelve forced family
replacements, but the independent validator exposed two additional stale
assumptions: it required `serial_fifo` despite the production process reporting
the required four-slot `slot_aware` scheduler, and it rejected legitimate zero
process-wired measurements from macOS `footprint`. The reformulated authority
requires `slot_aware` with at least four slots; preserves strict positivity for
RSS, physical footprint, and host-wired memory; and treats process-wired bytes
as a nonnegative measurement. Serial fallback and negative-wired mutations are
green. The complete captured runtime validates under this model-free contract;
the exact-lineage rerun remains the hardware seal.

Rev 72 seals that exact-lineage rerun at runtime commit `cb622acc`. One
production four-slot process completed thirteen phases and twelve forced
replacements across Qwen3.8 dense, Qwen3.6 MoE, Gemma4, and DeepSeek4. Every
activation used a fresh generation, exact family replay and semantic canaries
held, evicted artifacts had no live ownership, and the independent evidence
seal revalidated. Switch time ranged from 0.560691 to 4.838193 seconds with a
3.306021-second median, below the fixed 60-second product ceiling. The sealed
matrix is
`/opt/hf2q-evidence/universal-release-cb622acc-generative-swap/matrix.json`
(SHA-256 `86e6739c476b9a4c0fc3ea2e28dced0fca5650d4ca85a1f6ff6ea1d434026bd6`).

The next Gemma B.2 spike first falsified a literal shell continuation embedded
inside a `jq` program; the runner parser and a targeted model-free canary were
fixed before any model load. At exact commit `a02e62e2`, the corrected OFF-arm
warmup then exposed a real universal SlotAware liveness defect: four short
terminal requests entered, two completed, and two timed out at the exact
300-second watchdog. The worker had drained channel work into its private
`pending` deque, yielded after inline GPU admission, observed an idle scheduler
after the terminal seed released its slot, then blocked on the empty channel
without consulting buffered work. Anchors had published and held no scheduler
ownership, so they were exonerated. Qwen and DeepSeek shared the same unsafe
idle shape. The reformulated invariant makes all three generative workers
rerun admission whenever their private deque is nonempty and blocks on the
channel only when both work domains are empty. Its model-free burst test is
green and pins all three family call sites. At exact commit `444c9ff0`, the
fresh Gemma hardware rerun completed all four simultaneous cold warmup
requests on slots 0, 1, 2, and 3 and shut down cleanly, hardware-confirming
the liveness repair. The performance producer then stopped before its first
sample because a second literal line-ending backslash remained inside a
different single-quoted `jq` predicate. That is runner failure, not model or
performance evidence. The reformulated contract lexes the complete runner and
rejects every line-ending backslash while a shell single quote is open; the
second defect is removed, the positive and injected-defect canaries are green,
and a fresh exact-lineage performance run remains required.

That rerun at exact commit `2a1be9f9` reached all four lanes and then rejected
its first 128-row sample because each rendered Gemma prompt contained 168 cold
tokens. The producer had treated payload-word count as total rendered rows and
therefore omitted the measured 40-token template/request envelope. No sample
was accepted and no performance conclusion is drawn. The corrected producer
subtracts that exact-artifact envelope before request construction and tightens
both producer and independent verifier from the former ±25% bin to at most
four rows of target drift. The verifier reconstructs the exact payload word
count and suffix rather than accepting a matching prefix, and warmups enforce
the same calibrated bin. A dedicated +5-row evidence mutant and payload-shape
mutant must now fail; the fresh exact-lineage A/B remains the deciding spike.

The calibrated `d08d3e1f` rerun accepted all three OFF widths and proved the
512-row request landed at exactly 512 cold tokens. Its first ON width then
failed closed because no four-lane aggregate trace fired: four requests were
prepared concurrently, but the worker executed four scalar prefills. A
separate canonical-launcher probe proved the live process retained
`HF2Q_CROSS_SLOT_ADMIT=1` and `HF2Q_ADMIT_COALESCE_US=25000`, with the
incompatible DFlash xlen flag absent. The next smallest spike exposes the
worker-lifetime conjunction—requested policy, hybrid scaffold, hybrid route,
DFlash exclusion, effective enablement, and coalescing window—in one startup
event. Its pure policy truth table is model-free; the event decides which
capability operand, rather than admission timing, disabled the intended path.

The startup event at `3b8feb62` then proved every frozen operand true:
requested policy, hybrid scaffold, hybrid route, DFlash exclusion, effective
enablement, and the 25 ms coalescing window. Source tracing found the actual
route: ordinary Gemma chat is classified Stable with a measured 121-token
boundary and seven-token cue. The Stable planner incorrectly applied the
cold-prefill 32-row containment floor to that live cue and therefore forced
the four prepared lanes through scalar admission.

Two hardware falsifiers reject the obvious relaxation. First, an instrumented
working tree based on `3b8feb62` compared the four-sequence live-resume cue
against four scalar recurrent resumes at cue lengths 2, 3, 4, 5, 7, and 31.
The five-token case selected different tokens in lanes zero and one, so tiny
cue batching is not equality-preserving. Second, the reformulated split path
batched only an eligible 32-row boundary extension and replayed each tiny cue
through the proven recurrent path. Its four-slot Engine gate proved exact
64-row reuse and one ON route event per wave, but the first post-cue logit row
already differed from scalar by max absolute 1.9777224 and RMS 0.3709889; a
16-token greedy continuation then diverged in text. Disabling the tiled-live
attention branch did not restore distribution equality (max absolute
3.0703387, RMS 0.6399542), even though that particular 16-token greedy sample
remained token-equal. Both candidates are rejected: same first token or short
greedy text cannot substitute for coherent logits under sampling, grammar, or
tool constraints.

The next synchronized spike, on the same artifact and `3b8feb62` working-tree
lineage, compared two scalar 32-row forwards with one two-lane concat forward.
The gated `iter_g_a_bisect_offset` test passed one qualified invocation. Its
per-layer scan used `HF2Q_SYNC_PER_LAYER=1`; unlike the earlier unsynchronized
host checksum, it proved both lanes exact at the layer-zero input and different
at the layer-one input. Four separate layer-zero binary row captures then
proved, with `cmp`, that the pre-layer hidden row, input norm, raw Q/K/V,
normalized Q/K/V, and SDPA output were byte-identical for each scalar/multi
pair. The first unequal stage in both lanes was `attn_out`, immediately after
the quantized output projection. Residual, router, and MLP stages inherited the
drift while expert IDs remained equal. This falsifies attention, cache
addressing, embeddings, and Q/K/V projection as the first cause for this
32-by-2 cold shape. It identifies the concat-width output projection as the
first changed operator. The investigation-only arm selector and rejected
runtime candidates were removed from the landing diff after the measurement.

The next exact stage bisection retained the scalar 32-row output-projection
identity with zero-copy lane views. Layer-zero output-projection rows were
bit-identical in both lanes, as were the following residual and all three
pre-feed-forward norms. Raw dense gate, dense up, and router projections then
matched per lane. The dense GELU-multiply output matched, and a complete dump
of every token's top-eight expert IDs and routing-weight F32 bits matched.
Dense down also remained exact. The first changed bytes were the raw expert
gate/up `mm_id` output; expert down and weighted combine differed only after
that first failure.

The deciding Gemma implementation hypothesis is therefore smaller than a
general body or attention rewrite. Preserve aggregate raw dense/router work,
GELU, routing, and dense down. Execute expert gate/up through exact zero-copy
lane spans at the canonical `M=32` scalar geometry, preserving ID order,
top-k bits, native GGUF weight storage, and the existing `SharedPerToken`
route. Because the two expert projections reuse mutable pooled routing
scratch that GraphSession does not infer automatically, every lane after lane
zero requires an explicit encoder memory barrier before reuse.

That smallest candidate passed the pinned real-artifact B2 M32 oracle. Two
fresh scalar lanes and one fresh rectangular arm produced identical selected
tokens and every finite final-logit bit in both lanes. The rectangular arm
processed 64 suffix tokens in 103.8 ms (616.5 token/s); the two scalar calls
took 67.3 ms plus 58.3 ms, 125.6 ms combined. This is 17.4% lower wall time and
about 21.0% higher aggregate throughput for the exact oracle. The full log is
`/tmp/hf2q-gemma-live-o-gu-exact.log`, SHA-256
`6061c3052e29fd27f147274331315578bc8b2940f187336e6cf4433f41d588a2`.
Because final logits are exact, the earlier expert-down and weighted-combine
differences were inherited from gate/up; those operators remain unchanged.
The checked-in ignored authority
`gemma_live_rectangular_b2_b4_state_and_continuation_are_exact` then closed the
B4/state gate on the same artifact. It uses fresh scalar and rectangular arms,
noncontiguous selected slots surrounding a deterministically seeded canary
slot, and direct byte-vector comparison rather than hash equality. Both B2 and
B4 matched final logits/tokens, every selected slot's complete logically live
hybrid-cache bytes plus cursor/layout metadata, the full physical unselected
slot, and a subsequent one-token continuation. B2 suffix wall was 126.401 ms
scalar versus 111.944 ms rectangular (11.4% lower); B4 was 245.089 ms versus
194.658 ms (20.6% lower). The 1/1 authority completed in 7.88 seconds; log
`/tmp/hf2q-gemma-rectangular-b2-b4-authority.log`, SHA-256
`46435d5990bca8d6010c4fbf365554208f8d48e94abbff9cd2ba6e3022b8a8e4`.
The deciding remaining coherence gate is realistic multi-token unary/SSE/tool
serving; the checked-in B.2 product runner remains the performance authority.

Rev 83 expands that authority from the earlier M32 sample to the complete
stable-route production matrix. At clean source `3e5d35e1`, packed-V and
full-F16 KV each passed 29/29 cells: B2 and B4, rows per lane
32/33/57/63/64/65/95/127/128/129/255/256, deep-cache starts at 500 and 1,000,
and a start-1,008 sliding-window wrap. Every cell proved byte-exact cue logits,
boundary-anchor restore, final logical cache, continuation, and full physical
unselected-slot preservation. Packed-V completed in 946.98 seconds (log
SHA-256 `a6dc2033b0f8a3b7203a540d4c9a06da18dcdf324707328c812c32eb7f1b45f2`);
full-F16 completed in 1,306.47 seconds (log SHA-256
`bb6860747580616c906691886fffc51087ef8e9ee9d64fbaeb36849647dfeacf`).

The subsequent realistic product spike accepted no performance sample. Its
first OFF continuation carried the exact generated assistant tool call and
matching tool result, reused 1,320 cached tokens, and rendered 1,471 prompt
tokens: 151 uncached rows, not the nominal 64. The earlier 40-token calibration
predated the hardened tool-result workload and cannot describe this envelope.
That first receipt alone could not distinguish fixed envelope from per-word
tokenization, so a second fail-closed spike supplied the necessary second
point: one payload word rendered 105 uncached rows. Together, the 24-word/151
and one-word/105 measurements solve exactly to 103 fixed assistant-tool/result
tokens plus two tokens per repeated payload word. A 64-row continuation is
therefore impossible for this template. The producer and independent verifier
target nominal 128/192/256 cells using 12/44/76 words, which render
127/191/255 rows, retain the ±4 usage gate, and stay under the production
256-row ceiling. The calibration is bound to exact model identity. Their
model-free contract passes and rejects 29/29 mutations. A fresh exact-lineage
product A/B is required; no speed claim is inferred from either rejected
spike.

Rev 85 completes that exact-lineage A/B at source `d05d3d3f` and rejects the
former all-width policy without weakening its predeclared gate. All 48 waves
were semantically and operationally valid: eight order-balanced OFF/ON pairs,
four distinct long-history conversations per wave, two unary plus two SSE
continuations, 1,320 cached tokens per lane, equal realized suffix widths
127/191/255, zero OFF rectangles, exactly one B4 ON rectangle, canonical tool
results, quiet host, AC power, and Fair-or-better thermal state. Nominal 128
had median speedup 1.105686 with order-stratified 95% CI
`[1.097205,1.133760]`; nominal 192 had median 1.114093 with CI
`[1.097664,1.152047]`. Both clear the immutable lower bound `>1.05`. Nominal
256 had median 1.034930 with CI `[1.013205,1.070982]` and therefore fails.
The independent summary is
`/opt/hf2q-evidence/adr049-gemma-stable-d05d3d3f-20260826T130900Z/summary.json`
(SHA-256 `9018e6a2e0dfb6c9900fcdb8ff9a16d79d1aa1602d425277ac731197d8f4cf96`);
the manifest SHA-256 is
`136547742b2eee0dcd5531e372e7f2dc225e74de9a0a0e34834cd4f57a65ce96`.

Phase attribution explains the crossover. At nominal 256, the ON rectangular
boundary median was about 1,435 ms, while the four OFF scalar prefills summed
to about 1,422 ms: the physical rectangle no longer reduces prefill work.
Nominal 128/192 retain roughly 10–12% product-wave benefit. The reformulated
production policy caps stable rectangular boundary rows at 192 while keeping
the low-level 32..=256 primitive available as coherence authority. Wider
stable suffixes use the canonical scalar route. The next product gate keeps
the 128/192 `>1.05` confidence requirement and turns nominal 256 into an
explicit zero-rectangle semantic/no-regression fallback cell. Until that rerun
passes, rev 85 is decisive policy evidence, not final product acceptance.

Rev 86 completes the required exact-lineage rerun at source `1e10e752`. All 48
waves again passed semantic, cache, route, identity, power, thermal, and host
contention checks. Nominal 128 admitted exactly one B4 rectangle per ON wave
and had median product speedup 1.106132 with order-stratified 95% CI
`[1.099957,1.128067]`; nominal 192 likewise admitted exactly one B4 rectangle
and had median 1.108905 with CI `[1.097661,1.154915]`. Both retain the
predeclared lower-confidence requirement `>1.05`. Nominal 256 emitted zero
rectangles in both OFF and ON, retained exact normalized tool semantics, and
had median OFF/ON ratio 1.033734 with CI `[1.026559,1.040282]`, clearing the
predeclared scalar-fallback non-inferiority bound `>1/1.05`. The independent
summary is
`/opt/hf2q-evidence/adr049-gemma-stable-1e10e752-20260826T133112Z/summary.json`
(SHA-256 `9c43096f829227b7e58f4cadc47a917950a9848e5724dff69ee32336692fe8ab`);
the manifest SHA-256 is
`192fb85d9bd15045303a48d29e9aeb17abc6f4e923059f7725b565dd1e079e66`.
This accepts the Gemma stable-boundary serving policy: aggregate B2/B4 work
through 192 rows per lane and retain the canonical scalar route above it.

The shared-weight native quantized-matmul hypothesis is falsified for this
Gemma B.2 route. The primitive was published as `mlx-native 0.15.1` from exact
source commit `f92eb020c4d3f821700e648fbce15d6cf75c2cc6`; the crates.io and
GitHub release archives were byte-identical at SHA-256
`76ce4c8d5773c72554a98020aadd330e566792dd27f843451ac5dd567bb6b5dd`,
and an independently downloaded crate passed its locked all-features suite.
The real 19.5-GB Gemma Q5_K_M state gate then proved B2 and B4 final logits,
selected state, unselected-slot canaries, and continuation byte-exact with the
primitive active (30 eligible output-projection dispatches per width); log
SHA-256 `3d32a5e64f64fef03afd39695e175bb7a521db20c46e0b5cf40ee5d3f19fe919`.
The pre-registered interleaved `C-B-B-C-B-C` performance falsifier produced
canonical B2 samples `[109669, 109863, 109708]` microseconds and broadcast
samples `[109135, 108619, 108883]`, medians 109.708 ms and 108.883 ms: only
0.752% lower wall, below the required 5%. The test failed closed before B4;
log SHA-256 `80c1ccaa4e7f8c114140c6dc68f3f44b0f57541b1c051af7a9411be969dc5167`.
Because the primitive does not earn its production complexity here, the hf2q
dependency bump, broadcast branch, and spike-only tests are removed. Version
0.15.1 remains a valid published backend release, but hf2q stays on 0.15.0
until a surviving feature requires a new pin. The accepted canonical
per-lane rectangle and its 11.4% B2 / 20.6% B4 gains are unchanged.

Production attention is not a surviving candidate. The live
hardware bisection proved layer-zero SDPA exact, and source review found that
the existing global path already uses physical slot views while the sliding
path already stages each lane chronologically before reuse. A proposed batched
hybrid replacement had no known failing production case and its TQ scalar
oracle substituted hybrid-vector attention for the live dequant-to-F16 tiled
route. That candidate is removed before compilation. Attention opens only as
a performance hypothesis under a direct production-route falsifier covering
B=2/B=4, M=32, global/sliding-wrap, and TQ/F16 cache modes. It must match
output and selected cache bytes, preserve unselected-slot canaries, and beat
the current route by at least 5% in paired same-process timing; an exactness or
speed miss removes the production candidate. The deciding spike failed its
first equality cell: B2 M32, noncontiguous slots `[3,1]`, global D512 attention,
packed-TQ V, start 32/capacity 96. The one-dispatch hybrid output differed from
the current production TQ dequant-to-F16 tiled route, so the timing phase did
not run. The disposable module and registration were deleted; no production
attention change survived. Compile log SHA-256
`3762671ec26eb670f97b74ed66231c3cc76b6878b03fb75f530a9117c2ca92e2`;
equality-gate log SHA-256
`ecace251935ba6ab1bf192e5aafef19b1e9d889e9c4a2d40e0b318a8848dc68a`.

Only the smallest surviving rectangular slice that produces exact final
logits, full hybrid-cache bytes, one-token continuation, and multi-token
unary/SSE/tool parity proceeds to product performance measurement. A failed
exact gate removes the candidate; no tolerance or greedy-only exception can
accept it.

**Lane B gates:** B.0 byte-identity (cohort+concurrent-decode); cooperative receipt regime (≥5 alternating serial/cooperative pairs, sustained median faster, peak RSS recorded, independent receipt verification); thermal contract (Nominal start, continuous Fair-or-better, no gap >5 s, fail-closed); memory H3 (≤116 GiB peak beside the 100 GiB artifact); product ceilings unchanged (60 s cold / 15 s cached-automatic-SSE / 35 s tool-result — never widened); B4 decode-cohort gate re-pass; the two B.1 latency contracts.

### Lane C — Hygiene & methodology (ships with Lane A's first PR)

1. Correct the stale docs that actively misled this research: engine_qwen35.rs:169-170 ("capacity = 1" — live registry is byte-budgeted, capacity `usize::MAX` via `with_byte_budget`, :523-526); kv_cache.rs:127-128 (`n_v_heads=8`/"~60-90 MB" arena — real: n_v=32, ≈1.96 GiB); engine.rs:2300-2301 (qwen35 "501 short-circuit" — stale since 2026-05); lcp_registry.rs:781-783 (chunk_pos "in params_hash" — it is mangled into tenant_id); investigation_env.rs:553-556 ("~96 MB per 27B checkpoint" — real ≈149.6 MiB); load_info.rs:2151 fixture note (missing MTP layer → 6.25% admission undercount for Qwen3.8-27B).
2. Document (not fix, this ADR) the two budget hazards: two independent 5%-of-RAM LCP budgets (engine.rs:3595-3597, engine_qwen35.rs:523-525 — same `default_lcp_byte_budget()` instantiated twice); `HF2Q_KV_PERSIST` carries THREE meanings (path — serve/mod.rs:974; `"0"` disable — :4457-4485; `"1"`/`"on"` enable — kv_persist/families/gemma4_dense.rs:21, kv_persist/index.rs:9). State the registry end-state: the SerialFifo registry + disk hydrate is the *permanent restart-hydrate tier* until the A.7 follow-on ADR replaces it — documented scope, not a stub.
3. Import FreeToken's evaluation discipline into release evidence: report worst-case (tail) TTFT against client watchdog ceilings, not just means; an agentic-stability criterion (decode rate within a fixed % of single-turn under the N=4 workload); the A.4 strict-equality idle conservation audit; a reference-model invariant battery over the AnchorStore state machine (injected-mutation style — FreeToken's equivalent suite caught 17/17).

### Qwen execution ledger — rev 18

Model-free evidence from the implementation lineage beginning at `95d618c8`
and its rev-6 gate-hardening checkpoint:

- `qwen35_anchor_store` has an independent reference state machine and fourteen
  focused tests. The mutation battery rejects 17/17 injected corruptions;
  A→B→C/rewind removes B and C before branch X can write; pending state is
  affinity-invisible; eviction is positional keep-newest-K; accounting charges
  four committed payloads plus one pending payload exactly.
- `slot_anchor_restore_preflights_every_payload_before_mutation` was added as
  a falsifier before the restore refactor and failed against the interleaved
  implementation: an invalid final recurrent payload left the cursor rewound
  from 14 to 9. After the two-phase preflight/mutate refactor, the same test
  proves all cursors, recurrent bytes, conv bytes, and parity remain unchanged
  on validation failure.
- `logical_buffer_copy_does_not_retain_chunk_sized_parent` proves the
  speculative hidden row owns a fresh logical-size allocation after the
  chunk-sized parent drops. Store accounting includes token/logit capacities,
  every nested recurrent/conv allocation, cursor tables, and the detached
  hidden row.
- The SlotAware worker now stages captures as pending, publishes only after
  the retained cache/ledger commit, selects the deepest epoch-valid match,
  prunes descendants before the first divergent write, and clears the full
  store before hard reset after failed restore, poison, or cold reset. A.8
  fields emit in structured logs and Prometheus counters.
- The initial per-slot 768 MiB envelope was falsified by the Qwen3.8 N=16
  target: depth-4 is about 9.4 GiB while a default grant may be about 4.5 GiB.
  The worker now resolves one immutable aggregate anchor-owned grant at startup,
  derives artifact-and-N-specific committed depth, applies a live aggregate
  pending preflight, and reports partial pending availability explicitly.
  Qwen3.8 N=16 therefore remains a supported serving shape without memory
  overcommit, but its receipt must report observed depth and pending-capacity
  slots rather than imply depth-4/full-pending coverage.
- Independent review found and this revision fixes five proof/accounting
  defects: idle audit now binds physical target/MTP cursors to retained
  ledgers; scheduler fixed bytes no longer double-charge the former single
  anchor; invalid spec metadata no longer mutates store payload behind byte
  accounting; the gate requires old C to restore only cold/A and uses a
  SlotAware-specific spec counter; store control allocations are included in
  exact owned bytes. Pending capture never grows committed control storage;
  publication evicts before its push, and a deliberately tiny grant selects
  zero control capacity instead of allocating then idle-failing. The rejected
  oversized request is correctly classified as admission isolation. The
  real-artifact script separately arms a one-shot failure only after a
  successful non-empty GPU prefill slice and requires full-store invalidation,
  hard-reset recovery, a cold unique-boundary retry, and reuse after rebuilding.
  Its Qwen3.6/Qwen3.8 hardware receipts were open at rev 11.

Rev 12 tested a selected-boundary one-forward hypothesis on the
54,657,734,208-byte Qwen3.8 BF16 artifact SHA-256
`f30d9a6ea40ca3c5265d0996a460ad1474173c40c8e7f04c0b03caf6084c2cee`.
The timings below are retained as measured evidence of the available cost, not
as an accepted implementation:

- A 33-pair OFF/ON split-boundary sweep produced identical sampled normalized
  responses. Median TTFT improved 25.86% at 66 prompt tokens, 35.40% at 70,
  35.36% at 74, 34.68% at 82, 34.11% at 90, 28.80% at 97, then tapered to
  6.80%/5.47%/3.55%/2.33%/0.40% at 193/321/577/1,089/2,113 tokens. This is the
  measured effect of removing an internal duplicate forward, not a decode
  claim or a universal equality proof.
- A matched four-slot spec-OFF wave preserved all four output hashes and
  physical scheduler/body/head width four. Wall time fell from 10.307992 to
  8.78 seconds (14.8% less); summed decode throughput rose from 29.7958 to
  34.7756 tok/s (16.7%).
- The historical width-four AUTO lineage gate passed A→B→C, rewind to branch X,
  rejection of stale old C, three live siblings, cancellation, transport-body
  isolation (HTTP 413; not a model-context proof), injected post-admission GPU-slice failure
  (HTTP 500), cold recovery, rebuilt reuse, and active speculative anchor
  traffic. It reported configured slots 4, effective committed depth 4,
  simultaneous pending capacity 4, aggregate grant 5,221,833,932 bytes, peak
  owned bytes 473,856,856, 15 restore hits, 5 descendants pruned, and 145,466
  slot-aware speculative anchor tokens.
- The over-context arm now derives the live model context from `/v1/models`
  and rejects a supposedly overflowing fixture unless its requested size is
  actually greater. This closes the earlier vacuous 140,000-word attempt
  against a 262,144-token model, which was valid input rather than admission
  isolation.

Rev 13 source-route review falsified the one-forward equality hypothesis even
though those sampled responses matched. The candidate changes `M`, cache
representation, or route selection relative to split execution for dense and
quantized projections, fresh versus resumed full attention, recurrent versus
chunked DeltaNet, MoE gate/up/down, output-head projection, and MTP. The
runtime integration was removed. The associated native selected-state
primitive was kept only on its experiment branch; it has no independently
justified production consumer and was not published.

Rev 13 also falsified the exact scheduler-coalescing spike from A.5. The ON
logs prove both authoritative forwards executed before checkpoint staging;
the OFF logs prove staging occurred between them. All 15 normalized responses
matched. Median total-time deltas for suffix widths 0/2/4/8/12 were
+0.18%/+1.18%/+0.69%/-0.18%/-0.57%, respectively: noise to small regression,
not a speedup. Independent review additionally bounded its cost with delayed
cancellation and a live but not-yet-budgeted pending payload. The spike and
its environment switch were removed. Native selected-capture work was not
published: its only hf2q consumer belonged to the falsified enclosing-forward
path, so backend PR #40 was closed with the branch retained as experiment
evidence.

Rev 14 executed that deciding block-segmented spike. Six focused model tests
pass: the two-layer cold-TQ 33/8 route, a nine-layer K=8 interleave, warm peer
isolation, exact write-generation rejection after a third suffix advance, a
queued layer-4 projection failure canary, and warm nonzero MTP plus policy-OFF
independent-cursor behavior. They compare exact logits, detached semantic
rows, target/MTP cache bytes, boundary anchors, cursor/parity receipts, and
actual dispatch-route tuples. Native ignored gates also pass on the exact
Qwen3.8 Q4_K_M artifact below and Qwen3.6 APEX Q5_K_M SHA-256
`f2c702182a4661d2cef573b388ff23336ce65aabb112762d1c1a24d4ba0cbc25`.

The first independent production-route A/B used the checked-in
`bench_qwen35_compound_boundary_ab.sh` harness on an M5 Max with the exact
19.5-GB Qwen3.8 Q4_K_M artifact SHA-256
`1ee55c653644d6f645c6b2f39fc56a3ce28093620fd34dd43678875f348f2e1a`,
speculation OFF, TQ KV ON, terminal K=8, four physical slots, temperature 0,
and identical 2,113-token prompts split as 2,048 then stable-prefix 58 plus
generation tail 7. Across three single requests and three four-client waves,
all 15 candidate requests emitted the actual compound-route receipt and the
main/candidate semantic plus token-count receipt SHA-256 was identical
(`27f37665fefe092c42bc4d5a40ebfceff306cba905610ef05787373c1ab0f829`).
Median single-request TTFT changed from 5,325.031 to 4,953.463 ms (7.50%
faster), total wall from 5.375298 to 5.004268 s (7.41% faster), and the
four-slot wave from 19.3892 to 19.0030 s (2.03% faster). This is the deciding
spike, not the landing claim.

The measured result accepts the block-segmented hypothesis for implementation
and reformulates the remaining fail-closed gates. Before landing: reserve the
full retained payload plus the compound route's incremental owned-buffer peak
before any compound-specific allocation (the ordinary per-window Metal arenas
are already bounded and admitted identically for either route);
bind final anchors to their cache instance and source slot; validate token,
KV, logits, and speculative payload shape before staging; inject failures
after prefix-MTP mutation and at the engine publication boundary; compare
ordinary production output independently with main; and prove native codec
plus physical-width coverage in the dedicated matrix gate rather than
rerunning the full semantic workload per encoding.
Only after those gates pass does a larger matched run become publishable
performance evidence.

Rev 15 closed a proof-integrity defect exposed by that larger run before it
could publish a result. Exact main `6655f965` predates the scheduler/body/head
physical-width counters, but the first landing runner required those
candidate-era metrics from both arms. The run therefore stopped on baseline
trial one before producing an A/B verdict; this is instrumentation failure,
not performance or model evidence. The executable contract now requires both
arms to complete the same four concurrent client requests, records an old
baseline's complete absence of all eight required counter/gauge samples as
`physical_instrumentation: "unavailable"`, and rejects partial or duplicate
telemetry.

Rev 16 then falsified the assumption that this one-token prefill/TTFT workload
should itself reach physical decode width four. All four candidate clients
returned exact `READY`, but each prompt became decode-ready one at a time; its
single useful token plus stop completed before the next 65-token tail. The
honest counters were scheduler/body/head maximum width one, four scheduler
steps, four handles, four target forwards, and four rows. Requiring width four
would optimize the fixture by delaying a semantic first token, not prove the
compound-prefill benefit. This gate therefore records candidate physical
instrumentation and observed width without claiming decode batching; the
separate long-output physical multi-slot matrix remains the authority for that
property.

The same review found three more proof hazards and the runner closes all of
them: acceptance minima are immutable and recorded (1.01× single TTFT, 1.0×
single wall, 1.0× four-client wall); existing or nonempty evidence directories
are rejected and the schema-3 pass receipt is atomically published without
overwrite; and four fresh processes run in baseline/candidate/candidate/
baseline order with every raw timing sample retained. The executable fixture
is blocking CI. Both failed evidence directories remain retained, and neither
contains a publishable speed verdict.

Rev 17 ran the corrected schema-3 gate at exact candidate `d9370ca5` against
exact main `6655f965`, using the qualified Q4_K_M artifact above and two fresh
processes per arm in baseline/candidate/candidate/baseline order. Ten raw
samples per arm produced medians of 5,641.102 versus 5,218.499 ms for single
TTFT (7.49% lower candidate latency), 5.692077 versus 5.275830 seconds for
single wall (7.31% lower), and 24.019550 versus 21.783300 seconds for the
four-client wave (9.31% lower wall, or 10.27% higher wave throughput). The
semantic SHA-256 `498fe0ab70c4fb14f5825105cb076114eb2e064668855c8a1fa132342de9b36f`
matched across all 100 responses; the candidate emitted exactly 50 compound-
route receipts. All ten candidate waves honestly reported scheduler/body/head
maximum width one for the one-token response, so this result claims compound-
prefill latency and concurrent-client wall improvement, not physical decode
batching. The immutable receipt is
`/opt/hf2q-evidence/qwen38-compound-d9370ca5/receipt.json`; cross-codec,
cross-family, and matched current-reference gates remain independently
required.

Rev 20 ran the full SlotAware semantic-state gate on an M5 Max at exact
runtime commit `0da66164`, using the 19.5-GB Qwen3.8 Q4_K_M artifact above,
four physical slots, speculation AUTO, and a fresh 20,576-token opening
prompt. The immutable summary is
`/opt/hf2q-evidence/qwen38-anchor-0da66164-proof/summary.json` (SHA-256
`3c17ae0fadbb073b9b9811478b88a411a3876abbc880dcaed03c92a0b4789982`).
The gate passed A→B→C, rewind to branch X, stale-old-C rejection, four
concurrent clients, cancellation with three live siblings, exact model-level
context rejection (HTTP 400 at 300,052 tokens), and a one-shot failure after a
successful non-empty GPU prefill slice (HTTP 500). The failure cleared the
store and forced the unique boundary cold (`cached_tokens=0`); after rebuilding,
the same lineage reused 20,696 tokens. Across the run there were 12 restore
hits, five descendant prunes, and 226,958 speculative-boundary tokens restored.
Configured slots, effective committed depth, and simultaneous pending capacity
were all four; the 4,229,595,955-byte aggregate grant held a measured
631,971,868-byte peak. The earlier default 4,096-line fixture produced 163,936
prompt tokens and timed out after partially processing 151,552 tokens; it is
falsification evidence for that gate default, not an anchor failure. The
reproducible default is now 512 lines, and the overflow fixture stays below
one MiB so transport admission cannot masquerade as model-context proof.
The orthogonal native-format/physical-width matrix at exact commit `4d66f9bc`
passed Qwen3.8 BF16, Q4_K_M, Q5_K_M, Q6_K, and Q8_0 at physical widths
1/2/4/8/16 with exact scalar replay per lane; its receipt is
`/opt/hf2q-evidence/qwen38-physical-multislot-4d66f9bc/matrix.json`. This is
why rev 20 removes codec-by-width duplication from the semantic-state gate,
not why codec coverage is omitted.

Rev 21 ran the same product gate on the 25,043,007,488-byte Qwen3.6
35B-A3B APEX Q5_K_M artifact at exact runtime commit `0da66164`. This is the
40-layer, 256-expert/8-active MoE shape and is materially different from the
dense 48-layer Qwen3.8 cell. The first spike completed the ordinary lineage
work (12 restore hits, 247,133 tokens saved) but correctly falsified the gate's
assumption that every artifact carries speculative boundary state: the live
capability gauge was zero. The gate now requires positive speculative carry
only when `hf2q_qwen_anchor_spec_boundary_capable == 1`, and requires an exact
zero delta otherwise. The fresh corrected run passed the full four-slot
lineage/cancellation/context-rejection/post-GPU-failure sequence, reused 20,658
tokens after rebuilding from the forced cold recovery, pruned five stale
descendants, and reported full committed depth plus pending capacity at four
slots. Its summary is
`/opt/hf2q-evidence/qwen36-anchor-0da66164-proof-v2/summary.json` (SHA-256
`53d7a81b40be7264b88cbcfe5cadb8174a6ce04d6942168163f7bf1e80be3e99`).
The server mapped 25,031,311,360 native matrix bytes directly from the GGUF,
allocated zero anonymous matrix bytes, retained the conversion-time Q8_0
output head, and reported a 0.09-second load wall plus 19 ms warmup. The Qwen
family's real-model Lane A proof is therefore complete; Gemma4 and DeepSeek4
remain distinct state-engine proofs rather than Qwen gate variants.

Rev 22 then ran the family-neutral product gate on Gemma4. The full receipt,
native-storage accounting, corrected canonical warmup, and cancellation
reformulation are recorded in §A.6. This closes the Gemma state-engine cell
without claiming cross-slot checkpoint sharing or duplicating the state test
across behaviorally identical quant encodings.

Rev 23 ran the same product gate on the 100.05-GiB DeepSeek4 Q2 artifact. The
full state-engine receipt, native-storage accounting, validated hash-route
classification, and elementwise-state loader correction are recorded in §A.6.
This closes Lane A across every supported generative serving family.

Open proof work is now Lane B-specific: finish the tail-TTFT and append-only
no-regression receipts and execute Lane B's measured family-generalization
workload. Native GGUF codec and physical slot-width coverage stays in the
dedicated matrix gate, so the state-machine proof is not repeated for
behaviorally identical weight encodings.

## Falsified findings and open hypotheses (studied, decided, documented)

Framing (Robert): items below are either FALSIFIED with evidence in hand, or OPEN HYPOTHESES whose deciding spike is named — never silently parked scope.

| FreeToken concept | Verdict | Reason / data so far | Deciding spike or condition |
|---|---|---|---|
| q\* CPU–GPU co-execution | FALSIFIED (for this hardware) | One memory controller: CPU matmul adds no aggregate bandwidth to a BW-bound decode and cannot be bit-identical to Metal kernels (coherence gate). FreeToken's own degenerate limit (B_H→B_P ⇒ q\*→m) is this hardware's whole regime. | None on unified memory; the *method* (measured contended-pair bench → closed-form policy) stays reusable for genuinely disjoint domains. |
| Global LRU expert cache + elastic expert↔KV pools | OPEN HYPOTHESIS | Target models fit wired in 128 GiB today; no capacity misses exist. SSD-as-PCIe analog is ~5–7 GB/s vs their 25–53 — a harsher regime that the hypothesis must survive. | Spike: serve an artifact (or synthetic constraint) exceeding RAM through the deepseek4 mmap/residency seam (residency.rs:75-120) and measure page-in behavior + decode floor. Runs when a >DRAM model is targeted. |
| Elastic pool rebuild control plane | PARTIAL SPIKE: the worker primitive is source-coherent; manager policy and performance remain OPEN | A drained worker can release checked KV/prefix/scratch/capture ownership while retaining mapped weights. Receipts are not capacity credit because the current pool never charged those bytes. An indeterminate park acknowledgement permits only shutdown/full eviction. | Next: separately charge a nonzero configured mutable reservation, active-only lookup, reserve-before-reactivate, fresh generation publication, and full-eviction rollback for projector/KV-persist or non-cofitting entries; then matched A→B→A hardware proof. |
| bf16 chunk-pipeline state capture | FALSIFIED (for production capture) | The experimental FLA-style chunk path materializes per-chunk states like FreeToken's, but in bf16 — restore is not byte-identical (bf16 accumulation already failed the W-5b.3 walk-bar at pp4096). | An exact-F32 side channel is a legitimate new hypothesis, under its own ADR. |
| Dense 64-token stride anchors | OPEN HYPOTHESIS | See A.7. FreeToken itself only keeps the deepest crossed boundary per forward. | A.8 divergence-position histograms; decision ladder in A.7. |
| FTW weight format / graph capture | Not needed | hf2q already loads role-aware zero-copy GGUF into Metal buffers; mm_id grids are routing-independent so recorded-CB replay is routing-safe. | — |

## Sequencing & coordination (per AGENTS.md, which supersedes older session notes)

- All work in worktree branches under `/opt/hf2q-worktrees/`, PR to main; `/opt/hf2q` stays on main. Merges to main are **never blocked by gate runs** (identity checks assert main *ancestry*, not tip — `hf2q/release/protocol`); compile-quiet applies only during model-gate windows.
- Cross-platform coordination through Ruflo memory namespace `coordination`: check `hf2q/release/status` before heavy box work; `hf2q/release/driver` owns release dispatch; never two writers in one worktree.
- Collision map vs the active qwen38 universal-quant/multi-slot lane (which serves through qwen35 infra):
  - HIGH — the engine_qwen35.rs prefill-chunker region (`qwen35_next_prefill_end` :3635-3648, emission :3966-3985) and the engine.rs transaction-ceiling logic (16729-16749): `35e42b28` already widened single-slot to 4,096; the Lane A invariant *transaction end == stable boundary* must survive every further widening. Co-review any change to either.
  - MEDIUM-HIGH — anchor payload carries `spec`/`mtp_current_len`; `HF2Q_QWEN_SPECULATION` defaults to `auto` at HEAD (qwen35_speculation.rs:13-14, :55-57). Payload-shape changes co-review with the speculation lane.
  - MEDIUM (registry tier only) / LOW (anchor store) — KV-packing/quant changes: state-only anchors are packing-independent (cursors + F32 DeltaNet state).
  - HIGH — any future segmented-boundary path touches `forward_gpu.rs`,
    recurrent DeltaNet/convolution state, native output-head projection, and
    the MTP pending-hidden payload. Changes there must preserve per-segment
    route identity, terminal-state provenance, and whole-operation rollback;
    the split-forward path remains the authority.
  - Safe in parallel: kv_persist/* and unrelated artifact/accounting work that
    does not alter anchor payload ownership or admission grants.
- Order: Lane C + Lane A.1–A.4 (one branch, spike-first per the kata) → Lane A gates → A.6 parity branches → Lane B.0 spike branch (may run concurrently with A.6) → B.1 only after B.0 passes. Release-scope window for the first merge: next open minor after 0.1.10 (confirm with `hf2q/release/status`).

## Acceptance (the ADR is DONE when)

- [x] A.1–A.4 landed for the qwen35-family engine with the model-free battery
  and one real-artifact product proof for each served Qwen architecture shape
  (Qwen3.6-35B-A3B and Qwen3.8-27B; Qwen3.8 receipt linked above). Codec and
  physical-width coverage belongs to the separate native-format matrix rather
  than a duplicate semantic-state Cartesian product.
- [x] Family coverage complete per the scope directive: Qwen3.6/Qwen3.8,
  Gemma4, and DeepSeek4 are hardware-proven through rev 23.
- [ ] B.1 DeepSeek four-prefill-plus-live-decoders hardware gate passes
  parity, semantic-SSE/scheduler-tail, thermal, memory, and speed contracts.
- [x] B.2 Qwen aggregation and Gemma 128/192 aggregation plus 256 scalar
  fallback shipped with exact-result/coherence proof.
- [ ] Qwen's fixed-cost curve, rectangular speed bounds, Q5 timing cells, and
  matched-peer performance are reproduced under process-group-cpu-v2; the
  affected v1 timing receipts are historical evidence, not final uncontended
  authority.
- [ ] Gemma's accepted 128/192 product-speed bounds and 256 scalar fallback
  are reproduced under process-group-cpu-v2; the older v1 timing receipt is
  historical evidence, not final uncontended authority.
- [ ] The joined current-head Qwen product matrix seals Qwen3.8 dense/MTP and
  Qwen3.6 MoE/no-MTP rectangular ABBA, lifecycle/cache reuse, and live-decoder
  plus four-prefill Mixed cells from one exact binary.
- [x] Payload-ownership regression (no retained parent allocations) and the preflight-then-mutate `restore_slot_anchor` refactor landed.
- [x] A.2 lineage regression and the new SlotAware divergence gate exist in `scripts/` and fail closed.
- [x] Telemetry (A.8) emits one fixed-schema terminal outcome per restore attempt across Qwen, Gemma, and DeepSeek production paths; real-artifact telemetry receipts remain part of the family gates above.
- [x] Lane C doc corrections merged; budget hazards + registry end-state documented in operating-kv-cache.md; ADR-017-per-family-status row updated.
- [x] B.0 spike verdict recorded here (PASS; B.1 implementation candidate executed, hardware contracts still open).
- [ ] This ADR's Status flipped to Implemented (or Superseded-in-part with links), with Updated stamps at each landing.

## Consequences

### Positive
- Same-slot context edits stop going cold: restore from the deepest surviving
  family-native anchor, with one-row kernel capture reusing the existing
  linear-state storage class and no new weight representation.
- The Mixed-phase F-payment reduction attacks the measured 35 s tool-result failure mode at its actual location (projected 75 s → 31–46 s aggregate prefill work), instead of optimizing the already-working pure-prefill wave.
- The lineage/publication state machine, exact reclaimable accounting, and strict conservation audits raise the whole cache subsystem's verifiability — mutation-testable at AnchorStore scale before any shared-store ambitions.
- Falsified paths carry their evidence and open hypotheses carry their named deciding spikes, so future sessions extend the data instead of re-litigating from scratch.

### Negative
- An immutable aggregate host-RAM grant is available to anchors; actual depth
  varies with artifact and slot count. Qwen3.8 depth-4 is ~2.35 GiB at N=4,
  ~4.70 GiB at N=8, and ~9.39 GiB at N=16, so wider shapes may expose partial
  depth or pending-capture availability rather than overcommit.
- Anchor lifecycle adds state-machine complexity to the slot worker (three-state publication, epoch checks, descendant pruning) — the price of multi-depth reuse on a mutable KV log.
- Lane B carries genuine schedule risk even though B.0 passed: cache isolation
  is proven, but the real Mixed workload may still miss its scheduler/SSE tail,
  memory, parity, or speed contracts and require policy reformulation.

### Neutral
- FreeToken's remaining machinery stays un-imported; this ADR records why, which is itself a decision.
- The SerialFifo registry tier remains active for SerialFifo restore/hydrate but unused by the SlotAware scheduler; documented as the restart-hydrate mechanism pending the A.7 follow-on.

## Links
- ADR-040 (continuous batching; slot-aware scheduler substrate) — Status 🟢 full-context three-family workload served.
- ADR-042 (DeepSeek-V4-Flash; slice economics, cooperative cohort, receipt/thermal regime, product ceilings) — Accepted; Lane B lands inside its contract regime and must update it in the same work.
- ADR-017-per-family-status (KV persist family hooks) — Lane C updates its row.
- ADR-044 (qwen38 native; speculation default) — collision-map counterpart.
- FreeToken: arXiv 2608.16157; reference checkout /opt/freetoken (read-only).
- Review artifacts: Kimi K3 and gpt-5.6-sol full reviews in the research session transcript (2026-08-22); both verdicts and all MUST-FIX items incorporated above.
