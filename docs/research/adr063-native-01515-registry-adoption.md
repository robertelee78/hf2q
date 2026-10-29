# Published native 0.15.15 adoption and bounded DeepSeek ledger

hf2q 0.1.28 pins the normal-registry `mlx-native = "=0.15.15"` package,
SHA256 `bb508d36e311a24147e917282a8ccae0d47720f4a12ae678717df5d569c75e3c`,
from native merge `9c67e1fb670303aa5f30da9bbfc52fbe79652a4e` (PR #76).
[Native release run 40937109637](https://github.com/robertelee78/mlx-native/actions/runs/40937109637)
completed successfully. Independent registry/index/VCS/tag/GitHub-asset/sidecar
verification matched the exact source. All 684 archive members also matched the
actual Cargo registry source used to compile this consumer. No local Cargo patch
or unpublished dependency is used.

The dependency adds bounded same-file GGUF header parsing and a device-free map
plan using the production coalescer. The cfg(test) DeepSeek diagnostic consumes
that API without model loading, mmap, payload reads or device creation. Geometry
remains an explicit caller-assumed projection. Logical file extents, rounded
registrations, source F32 expansion and cache projections remain separate.

## Validation

Executed consumer source `b37393f4128d6ef44cdd912f3b1dc4d23c4d7d71`:

- `cargo check --locked --all-targets --all-features`: PASS, 35.46 seconds.
- `cargo build --release --locked`: PASS, 42.21 seconds; binary reports 0.1.28.
- Six bounded-ledger controls: PASS, including exact tiny catalog, separate F32
  expansion, file extents, config shape and storage-role boundaries,
  regular/no-follow input and source/output quotas.
- Nine supervisor parser/guard controls: PASS.
- One separately supervised, metadata-only invocation on the retained approved
  DeepSeek-V4-Flash-0731 Q2 artifact: PASS; 0.523071 seconds, terminal checks
  0.941017 seconds, healthy owned process and empty reaped group/TMPDIR.

The full suite subsequently passed at
`3386bacea199dd05e8116cbbcacea4ed25f72865`:
`cargo test --locked --no-fail-fast -- --test-threads=1`, exit zero, 318.46 seconds.
Its 63 successful summary lines sum to 5,983 passed and 174 ignored (including
nested child summaries; not a unique-test count). Two test-only corrections
preceded this pass: an old reasoning golden was aligned with the already-merged
native JSON insertion-order contract, and a static-info test received a private
state root. Production behavior and all reasoning/verdict assertions were kept.
The original failed logs remain retained. The bounded diagnostic source and
normal production paths are unchanged by those two corrections.

## Real-artifact metadata result

All 1,328 required tensors were present with exact catalog/shape/storage matches;
there were no extras. Actual bounded header reads were 5,336,594 bytes, declared
parser reservations 15,535,724 bytes, 386,430 array elements and nesting depth one.
Two planned native roots total 107,426,037,760 requested bytes. Required owned F32
expansion adds 7,518,556 bytes: full requested weight upper 107,433,556,316 bytes.
The registered file-page union is 107,426,021,376 bytes, with 16,384 bytes overlap
and a 2,912-byte final EOF tail. Neither overlap nor tail reduces the root charge.
One C262144 cache projects 1,821,392,896 requested bytes; actual owner count and
bootstrap eligibility remain unobserved here.

Evidence bundle `q5-ds-bounded-map-ledger-actual-v1` retains raw descriptors,
bindings, process observations and cleanup. Independent artifact review checked
all 1,328 descriptors and sums. The actual source/binary were the executed b373
checkpoint; later test/documentation commits are not relabeled as that binary.
The cached file's retained full-hash receipt and unchanged no-follow stat were
joined without a new full-payload hash or download.

This is metadata evidence. It grants no physical credit, model residency,
serving/quality/performance qualification or largest-profile admission. Full
four-family and Q5 acceptance remain pending. Existing ADR-061 release rules
retain model qualification as ad hoc rather than an automatic shipping gate.
