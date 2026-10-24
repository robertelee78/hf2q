# Optional V2 bounded meminfo calibration

This is a diagnostic instrument, not a qualified observer or admission method.
Its one bounded V2 trial completed with an inconclusive idle-noise outcome, as
recorded below. Import and pure fixtures do not load libc, query hardware or
allocate the payload. Runtime admission, source-age proof and whole-memory-bound
flags remain false for every outcome, including observed responses.

V1's single stopped trial falsified the assumption that the CSV total always
matches installed physical capacity: the human summary explicitly reported an
Unaccounted21page residual omitted from the CSV total. Its first six accepted
samples also spanned23,707,648bytes, above the fixed8MiB idle range. V1 and its
abort are preserved separately; V2 does not repair its noise or relabel its guard
as passed. No V1 branch code is a prerequisite for this additive candidate.

V2 supports a narrowly corroborated interface. For the same invocation, require
six nonnegative integer CSV categories summing to CSV subtotal T; independent
physical capacity P; T<=P; and residual U=P−T an exact whole number of host pages.
The formatted Physical Memory header's integer pages must equal P, and exactly
one top-level row for each named category must match its CSV bytes exactly.
For U>0 exactly one top-level Unaccounted row must explain the entire residual;
for U=0 it may be absent or explicitly zero. Duplicates, malformed grouping,
unknown roots, nested/fake/Swap substitutes, missing terminator/header, warnings
and contradictions reject. Rounded MB/percent/bar values never supply accounting
values. The intentionally strict grammar may reject a new system output format;
it must not silently broaden to recover a result.

Receipt records P, csv_accounted_subtotal and unknown_residual_bytes separately,
plus the parsed page tree. Occupied=P−Free includes U in full. U is unknown
classification, never available/reclaimable/cache/reserve credit. CSV total is
not silently renamed physical capacity. Input stem and raw CSV/stdout SHA256
bind the pair from one invocation; malformed pairs never enter the schedule.
The source and parser hashes are separate receipt fields.

A future trial requires independent source/fixture review, an immutable freeze,
an external60second process-group watchdog, and a recorded agreed quiet local-tool
window before any trial. The required CLI attestation describes that window and
how unexpected activity will be recorded by the supervising controller. It does
not assert OS quiescence or authorize app quitting/configuration changes. No
builds, other local probes or competing tool work should be scheduled in that
window. Unexpected activity remains in the record; no selective sample deletion
or automatic retry is allowed.

The fixed64 serial samples target250ms spacing within each phase; the schedule
restarts after each phase's materialization/proofs rather than maintaining one
global250ms cadence. Phases are I0 idle12, A16 anonymous16MiB8,
A32 anonymous32MiB8, F16 private-file16MiB8, Fhold unchanged8, Aoff anonymous
unmap8, I1 file unmap/close/unlink12. Maximum explicit payload48MiB plus bounded
page/status/CSV buffers stays below64MiB; it is not a whole-process bound. No
models, GPU, privilege, purge, global sync, fsync, payload/threshold escalation or
retry. The file is written in page chunks then mapped shared readonly; dirty/file
cache is never assumed instantly reclaimable. Before/after plateau mincore
residency plus page-sized deterministic readback independently check own mappings.
Residency failure is inconclusive, not silently retouched. Readback may affect
later residency; its ordering and timestamps are retained. No whole-map copies;
cleanup failure invalidates the result and payload files are not evidence hashes.

Installed SDK sys/mman.h SHA256:
`07adbf7f11a601e6461ca508a2deca037103008600608f8dbe93a4b0887d46e9`.
ABI: mmap(void*,size_t,int,int,int,off_t), signed64 Darwin off_t;
munmap(void*,size_t); mincore(const void*,size_t,char*). READ1/WRITE2,
MAP_SHARED1/PRIVATE2/ANON0x1000, MAP_FAILED(void*)−1, MINCORE_INCORE1.
Independent capacity comes from hw.memsize at execution only. meminfo binary is
pinned before/after to SHA256
`a7bf5894154086eb8d48f4232a18053d666369132bb221b7cd1d5b2abcfa7cdf`.
The prior runtime platform-bit observation is separate provenance evidence, not
a guarantee of atomic counters, source age or responsiveness.

Command timeout1s, accepted latency/lateness each<=100ms. CSV timestamp must fall
within invocation wall brackets plus100ms tolerance; wall/monotonic offset change
>50ms invalidates timing. Idle I0, Fhold and I1 occupied range and first/last-four
median drift must each be<=8MiB. Each anonymous16MiB step requires last-four median
occupied rise>=8MiB and first threshold-crossing invocation start within1s after
completed touch (including verification time). Both steps must pass. U series,
raw category changes and all samples stay visible; U has no new adaptive tolerance
or post-hoc instability exclusion. File/unmap curves remain diagnostic without
mandatory immediate free recovery. Unrelated activity can mimic or obscure the
signal even in a locally quiet window.

Outcomes observed/threshold_failed/inconclusive retain explicit reasons. Observed
means two responses in this host trial only, never causal attribution or universal
freshness. No sum of overlapping details, logical uncompressed data or disk swap;
no cached-file credit or PID subtraction. P−Free is not an exact disjoint C ledger,
future-growth allowance, complete process B, production fit or fullQ5 admission.

Pure controls (not executed by author):

```sh
cd scripts/four_agent
python3 -m unittest -v test_meminfo_interface test_meminfo_calibration
```

Synthetic fixtures model the observed21page shape without private captures.
Only after independent review and an external watchdog may an explicit trial use:

```sh
python3 scripts/four_agent/meminfo_calibration.py --run --output NEW_PRIVATE_DIRECTORY --quiet-local-tool-window 'Recorded controller window and unexpected-activity policy'
```

## Single V2 trial result — 2026-10-24

The integration owner ran one reviewed, frozen trial in an agreed local-tool
quiet window. Other local agent commands were paused; desktop activity remained
uncontrolled. The independent postexecution review verified all64 samples and
their input bindings. No retry, threshold change, sample deletion or rescoring
followed. Supervisor exit2, no timeout, elapsed14.865076959seconds; outcome
`inconclusive`, reason `idle_noise_I0`. This is not a responsiveness pass.

All64 physical interface checks reconciled: independent P=137,438,953,472bytes,
page size16,384bytes, six-category subtotal T and explicit residual U corroborated
by same-invocation integer page rows. U was zero in63 samples and147,456bytes
(9pages) in one. Occupied=P−Free included U in full. These observations support
only the interface instances actually checked, not atomic sampling or universal
accounting behavior.

All14 before/after materialization proofs were present: empty I0/I1, anonymous
16MiB A16, anonymous32MiB A32, anonymous32+file16MiB F16/Fhold, file16MiB Aoff.
Each mincore resident count matched the intended pages and same-kind/same-length
readback checksums were stable. This does not establish continuous residency
between checks. Cleanup errors were empty and no payload residue remained.
Ordinary interpreter/tool/OS overhead remains outside the48MiB payload claim.

| Idle phase | Occupied range (bytes) | Absolute first/last-four median drift (bytes) |
| --- | ---: | ---: |
| I0 | 55,427,072 | 3,301,376 |
| Fhold | 1,458,176 | 843,776 |
| I1 | 51,331,072 | 368,640 |

The fixed8MiB limit is8,388,608bytes. I0 range failed, so evaluation returned its
first inconclusive reason before assessing anonymous transition qualification.
I1 also exceeded the range limit diagnostically. Fhold's smaller variation does
not justify discarding other phases or establish causality. No pressure, stale
counter or desktop-activity cause was identified.

Invocation latency median27.6136665ms, maximum29.582875ms; maximum scheduled-start
lateness10.340292ms. These observed brackets are below100ms, not internal source
age or future maximum latency. Within-phase adjacent invocation gaps ranged
240.587417–260.247166ms. Phase-boundary gaps ranged47.665250–91.724667ms because
the next phase restarts its schedule after operations/proofs. The completed run
is therefore not a globally spaced16second series.

Exact SHA256 evidence identities (private raw evidence retained outside Git):

| Evidence | SHA256 |
| --- | --- |
| Pre-execution seven-file freeze | `eee2292dc2daaa22fbf27a10c33c58d5891d5b18ddb41a278e44cfea39a2bd96` |
| Controller source | `445c01d9033e0cd4e1d040dcbf7bf3275a2fa7b38de6b0ec531cfa235355246f` |
| Interface source | `78de91ab8ab8fb4ff8c0290ae844c9772cc0c1242afcf766e64beba361671e13` |
| Trial receipt | `83d6b0577ac4d2e66287145f9027af4db14f36591cc91ef0c598f847118d7e3b` |

The independent reviewer verified all256 artifact hashes and the archived source
freeze. This results-only note update leaves the six code/fixture files unchanged;
the pre-execution documentation hash remains historical. V1's guard failure and
noisy prefix remain preserved on their separate branch. Neither trial qualifies
a memory observer, source-age bound, admission method, model fit or fullQ5 result.
