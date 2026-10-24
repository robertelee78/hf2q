# Q5: bounded warmup plan controls

The existing fresh DeepSeek warmup runs 64 EOS tokens and its output-head probe.
This change leaves that operation, its production capacity and all allocation
policy unchanged. It extracts the existing prefill-span arithmetic into the
cache plan so metadata-only tests exercise the same function as the live cache.
The live wrapper still checks poison first, and the start-zero guard is unchanged.

For the frozen ungrafted 43-layer configuration, capacity 128 plans 18,722,816
bytes of cache payload versus 919,617,536 at capacity 131072. Both plans produce
the same fresh 64-row transaction span: 64 window rows, 16 ratio-four groups
and zero ratio-128 groups. The tests assert these absolute counts and ranges,
plus a closed list of permitted physical-capacity differences. They also show
why capacities 64, 126 and 127 have insufficient window or compressed-storage
shapes even though the plan constructor accepts those contexts.

Four metadata controls passed on Apple M5 Max with locked dependencies. The
complete cache regression filter passed 22 tests, including those four controls,
covering the existing live planner, poison, publication, snapshots and growth.
No model was loaded. The new controls create no device or Metal resources; the
existing cache regressions exercise bounded synthetic buffers. Source and log
hashes are retained in private evidence receipts.

These are necessary plan constraints, not proof that every encoder accepts the
smaller parent shapes or that numerical output is unchanged. The internal pure
planner retains the live path's validated-plan/graft assumptions; its checked
start overflow test does not establish validation of arbitrary forged plans or
every integer input. Nonzero cursors, grafts, longer or single-row probes and
configuration changes need their own evidence.

A later proposal may use a temporary 128-position cache for the fresh probe,
but it requires native encode checks and numerical parity before changing
production behavior. This change supplies no model-memory admission, physical
footprint measurement, full Q5 acceptance or speed claim.
