# ADR-063 Q5: host metric availability finding

Status: observed availability limitation; no calibrated memory policy or runtime approval.
This supplements the [Q5 design](adr063-q5-memory-design-2026-10-23.md).
The model-free ledger slice remains proposed. No model was loaded and no system
setting or privilege configuration was changed for these observations.

On the M5 Max / 128 GiB host, the unprivileged command
`footprint --all --noCategories --format bytes --json OUTPUT` exited 77 with
`footprint: Must run as root.` It did not create the requested JSON output.
The retained log is 29 bytes, SHA256
`cf0723b17d8d267b0090dc904731476cdeac9e3a46592bae88f6495bdd1d6fbc`.
An unavailable measurement must not be interpreted as a successful empty sample.
An independent review identified that this stderr-only log did not itself bind
the invocation or exit code. A second read-only attempt retained structured argv,
caller UID, timestamps, exit 77, stderr and absent-output status: 536 bytes,
SHA256 `d4ed1d9bbb26dd609350283fd596688515a7c1b9c7083486fe42c9cb82e45bb7`.

A separate read-only `proc_pid_rusage(RUSAGE_INFO_V4)` availability sweep used
the installed SDK field layout through Python ctypes. Of 1,049 enumerated PIDs,
676 same-UID calls succeeded; 372 other-UID calls returned EPERM and one returned
ESRCH. Its private receipt is 193,449 bytes, SHA256
`fa271dc4f45500cb9cbc9600c9acb2ef2bc094eaa172caf2269f424b30e113d3`.
The SDK-derived structure size was 296 bytes. A compiled ABI/offset check and
metric-semantics calibration remain required; this sweep proves only observed
API availability, not reliable complete competing-memory accounting.

“External user processes” in the design means competing user-space workloads
regardless of UID. Other-user, root-owned and service-account workloads cannot
be omitted because their metrics are inaccessible. Nor does the minimum 12 GiB
reserve prove that all denied processes fit within it. Partition candidate
ownership, harness/sampler, competing workloads and a reviewed OS allowance
explicitly, without omission or double charging. Preserve PID plus process-start
identity across enumeration races; EPERM is not an ordinary exited-PID race.

Complete competition therefore remains Unknown and runtime admission rejects.
Two possible follow-up designs require their own evidence:

- A narrowly scoped external privileged observer, keeping the model engine
  unprivileged. First prove coverage, metric semantics, bounded sampling and
  watchdog behavior. Root access is not yet proven sufficient for all targets.
- An independently established conservative bound for every unseen partition,
  assigned once to competition or the OS reserve and protected against growth.
  Same-UID sums, an idle snapshot and global free-memory percentage do not
  establish that bound.

This finding does not authorize sudo, a privileged helper installation, a daemon,
sysctl changes or a weaker admission rule. It does not block pure schema and
arithmetic work: fixtures should reject same-UID success plus denied external
rows, missing output, stale observations, PID reuse and incomplete enumeration.
The remaining ownership/scratch bounds, working-set policy, watchdog calibration
and separate runtime decision are unchanged.
