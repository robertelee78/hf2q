# Native 0.15.2 registry adoption

Date: 2026-10-24. Current hf2q source adopts exact registry dependency
`mlx-native = "=0.15.2"` for the mapped-buffer backing lifetime correction.
This is a downstream dependency adoption, not a new hf2q published release.
The source package version remains 0.1.27; the historical published 0.1.27
stock package and its native 0.15.1 identity remain unchanged.

## Independently verified published identity

- Registry archive SHA-256:
  `e09c782f1b774876674d04c15f03fae2f81b58dfe6401aba87153dcbe9d3a3a4`.
- Clean embedded VCS source:
  `7bb30a2fac783e0b340ec2f2086289f9751dda7a`.
- [Native v0.15.2 release](https://github.com/robertelee78/mlx-native/releases/tag/v0.15.2).
- [Exact-package release workflow, run 40070484734](https://github.com/robertelee78/mlx-native/actions/runs/40070484734).
- [Registry package](https://crates.io/crates/mlx-native/0.15.2).

The independent verification receipt confirms the registry archive, GitHub
release asset, checksum sidecar, tag/source and completed release workflow
agree. It records `ready_for_hf2q_pin=true`. Verification also ran all eight
mapped-backing controls from the downloaded registry archive on the operator's
Apple M5 Max: eight passed, zero ignored. The archive tests checked ordinary
views, raw Metal/recorded owners, escrow-safe backing retention and teardown,
invalid ranges, deterministic nil construction and unwind cleanup. They used
tiny buffers without model loading, GPU command encoding or tensor payload
reads after drop. No model quality or speed result is implied.

## Downstream contract

The exact Cargo requirement and registry lock must resolve this archive; no
path/git replacement or ignored local patch is evidence of adoption. Live
dependency verifiers, release receipt producers/consumers, matched-reference
bindings, execution trace checks and source-teacher producer labels now use
0.15.2 and the verified checksum where applicable. Historical/frozen campaign
receipts and version-independent synthetic identity fixtures keep their
original identities. A later campaign must freeze its actual new source-built
binary and dependencies rather than edit old stock or F1 manifests.

The correction retains each mmap through the no-copy Metal resource's captured
deallocator block, including independent raw/recorded resource owners. Extra
Arc ownership is not another mapping; copied block/descriptor host overhead
still belongs in complete memory accounting. Public signatures, tensor format,
kernel arithmetic, model policy and logical resource sizes are unchanged.

The narrow archive controls do not establish complete Q5 ownership/scratch
bounds, OS metric coverage, full-model admission, tool/SSE correctness, a
measured performance win or the all-four speed verdict. Source/build/package
compatibility and focused downstream gates remain part of the pin change's
validation; this record does not claim those checks passed before they run.
