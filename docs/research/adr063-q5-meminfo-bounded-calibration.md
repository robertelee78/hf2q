# Optional bounded meminfo calibration

This is CPU observation infrastructure, with no admission, universal counter-age,
cache-reclaim, model-fit or performance claim. Importing its module and running its
pure tests does not load libc, query hardware or allocate the calibration payload.
Execution requires `--run` and a new evidence directory. Independent review and an
external 60-second process-group watchdog precede any observation run.

The fixed schedule has 64 serial one-shot `meminfo --summary --csv PATH` samples at
250 ms target spacing: I0 idle12, A16 anonymous16MiB8, A32 anonymous32MiB8,
F16 additional private-file16MiB8, Fhold unchanged8, Aoff anonymous unmap8,
I1 file unmap/close/unlink12. No retry, escalation, purge, global sync, fsync,
privilege, models or GPU. Maximum explicit payload is48MiB plus small page/status
buffers, below64MiB; this is not a whole-process allocation bound. File bytes are
written once in bounded chunks then mapped shared readonly, with no clean-cache
assumption. Anonymous pages receive a nonconstant page pattern.

Independent own-map checks use libc mincore then page-sized deterministic readback
before/after each plateau. Partial residency is inconclusive, never silently
retouched. Readback may itself affect residency after the preceding observation;
the receipt preserves both timestamps. There are no whole-map copies. Mapping
addresses remain private and are invalidated after successful munmap. Cleanup
failure invalidates the result, and payload files are never hashed/copied as an
evidence artifact. Retain private evidence locally, not in Git.

ABI reference is the installed MacOSX SDK `usr/include/sys/mman.h`: protection
READ1/WRITE2; MAP_SHARED1/PRIVATE2/ANON0x1000; MAP_FAILED(void*)-1;
mincore(const void*,size_t,char*), MINCORE_INCORE1; mmap offset is Darwin off_t
(signed64). Exact SDK/header identity belongs in the execution receipt alongside
the machine/OS identity; constants are not inferred from a different platform.
Physical capacity is queried independently through hw.memsize at execution only.
The author-inspected sys/mman.h SHA256 is
`07adbf7f11a601e6461ca508a2deca037103008600608f8dbe93a4b0887d46e9`.

The parser requires one row, unique headers, all six physical categories and
total, nonnegative integral bytes, exact six-category reconciliation and matching
physical capacity. Additional overlapping fields are ignored, including logical
uncompressed data. Disk swap is excluded. All cached files stay occupied; no
PID footprint is subtracted. Current occupied=P-free is not a disjoint exact C.

Per-command timeout1s; accepted invocation latency and target-spacing lateness
each<=100ms. CSV wall timestamp must lie within the invocation wall bracket with
100ms tolerance. A wall/monotonic offset change>50ms invalidates timing. Idle I0,
Fhold and I1 must each have occupied range and first/last-four median drift<=8MiB.
The two anonymous16MiB steps must each show last-four median occupied increase
>=8MiB and first threshold-crossing invocation start within1s of completed touch.
Materialization verification time is included in that lag. Record failures and
all raw categories; unrelated activity can mimic or hide a response. File and
unmap curves are diagnostic only, without mandatory immediate free recovery.

Outcomes observed/threshold_failed/inconclusive are fixed, with explicit reasons.
Observed means two responses in this trial only. Runtime admission, source-age
proof and whole-memory-bound flags remain false. Provenance/schema/materialization
faults also remain nonqualifying; the reason identifies their class. The binary is
pinned to SHA256
`a7bf5894154086eb8d48f4232a18053d666369132bb221b7cd1d5b2abcfa7cdf`,
checked before/after. Its prior observed runtime platform bit is separate evidence,
not a guarantee of this run's freshness or atomic accounting.

Pure test command (not executed by author):

```sh
cd scripts/four_agent
python3 -m unittest -v test_meminfo_calibration
```

Optional runtime command must be externally supervised, after review:

```sh
python3 scripts/four_agent/meminfo_calibration.py --run --output /new/private/evidence-directory
```

No observation or pure test result is claimed by this initial implementation.


## Stopped v1 trial — 2026-10-24

The one frozen v1 trial returned inconclusive during I0, before any anonymous
or file payload mapping. Six samples passed; the seventh CSV selected-category
sum and CSV total were 137,438,609,408 bytes, versus independent hardware capacity
137,438,953,472 bytes. The same invocation's formatted summary explicitly showed
21 Unaccounted pages (344,064 bytes at 16 KiB/page) and the full physical total.
This disproves the exhaustive six-category CSV assumption for this interface.
It does not establish stale counters, a capacity change or a particular cause.

The strict guard correctly stopped; cleanup reported no errors or payload
residue. The first six occupied observations also ranged by 23,707,648 bytes,
above the fixed 8 MiB noise threshold. No responsiveness phase ran, and ignoring
the residual would not qualify this prefix. No thresholds were relaxed or trial
repeated. Source and raw evidence remain preserved; this v1 spike must stay off
main.

Tested script SHA-256: `f198472229f4eab2c5ed5d7ae1ffa0cc006d2efa5fd22f2b107d96db32cdcec4`.
Trial receipt SHA-256: `62cb0c435042499587ea2bc93706418ebb36118d6990a31f443a7a9e77158f56`.
A new protocol must corroborate any explicit unknown bucket rather than discard
it, retain all unknown bytes as occupied, and receive fresh review before another
trial. Runtime admission and source-age qualification remain false.
