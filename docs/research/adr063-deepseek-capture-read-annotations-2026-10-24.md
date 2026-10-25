# DeepSeek captured resource read annotations

Updated: 2026-10-24. Source basis hf2q
`017ac49d5c608a86ef448f6695dbb5a39e680d70` and published mlx-native0.15.2.
Root validation is recorded below.

The five uncompressed attention RMS calls, five compressed-core RMS calls and
two FFN RMS calls previously omitted their bound GPU parameter buffer from
captured reads. The ratio-four indexer query RoPE similarly omitted positions
and frequencies. These nodes retained nonempty read/write lists, so the native
empty-annotation check could not detect the omissions. This is a source-backed
metadata gap, not an observed corruption claim.

Three family-private annotation-and-dispatch helpers now include those actual
operands immediately before the unchanged native RMS/RoPE operation. All thirteen
calls consume the helpers. Numerical arguments, kernels, allocation, graph order,
default grouping, mode selection, cache behavior and dependency pin are unchanged.
Other native dispatchers' internal per-node status/map/indexer annotations remain
intact. Extra read metadata can conservatively constrain legal reordering; no
speedup or full phase-reuse safety is claimed.

The ignored exact control
`prefill_annotation_actual_capture_covers_parameters_and_rope_reads` invokes all
three production helpers and inspects actual CapturedNode buffer object/offset
bindings and read/write coverage. It uses eight distinct roots with64-byte guards
on both ends: input512B, weight512B, two RMS outputs512B each, parameters8B,
positions4B, frequencies128B, RoPE output256B. Logical payload2444B plus1024B
guards totals3468 requested Metal bytes. No model/weights or payload files are
read. Scoped CPU numeric data is below4KiB; no output matrix copies exist. Shape,
captured graph, pipeline/compiler/driver and allocator metadata are excluded from
that numeric bound and remain unqualified.

Inputs are initialized before capture. The test records exactly three dispatches,
extracts the capture, drops the unsubmitted session, and never finishes, commits,
reorders or replays it. It asserts the process commit counter unchanged. Captured
Metal owners and original roots stay live through binding inspection; guards are
checked without GPU work. Creating the device, allocations and pipeline compilation
still require Apple hardware and separate execution review despite no GPU submit.

Four cloned-node negatives remove the actual parameter read for each RMS and the
position/frequency reads separately for RoPE. Actual binding identities remain
intact and read lists stay nonempty. The ordinary completeness count stays zero,
while the binding-range oracle rejects each missing read. This validates the
oracle against omissions rather than asserting a mirrored expected read list.
It is annotation coverage, not numerical GPU parity or a before-fix runtime fault.

The tracker uses conservative whole-root CPU virtual ranges. This covers views
of the same Metal root but does not identify physical aliases through distinct
mappings. Correct annotations alone do not authorize scratch phase reuse, CPU
input recycling, decode multi-command reuse, returned-output reuse, accounting
credits, full-model speed/quality or ADR063 Q5 closure. Those remain separate.

## Validation

The exact ignored capture control passed once on Apple M5 Max: one matching test,
one passed, none failed or ignored. It verified three actual dispatch bindings,
four missing-read negatives, guards and unchanged commit counter. The supervisor
reported unchanged frozen source, no timeout/signal/cleanup error, and the exact
completion marker. Independent postexecution review verified the archived source,
binary, raw log, receipt and supervisor hashes. No GPU submission or numerical
parity experiment was performed.

Locked all-target/all-feature check passed (27.94s), and the locked release build
passed (36.48s). These checks establish the annotation repair within its stated
scope; real-model, serving and performance validation were not run.
