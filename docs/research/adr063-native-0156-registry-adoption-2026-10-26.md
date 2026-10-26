# Native0.15.6 exact registry adoption

The native release workflow40399608098 completed successfully from main commit
`ba0e3bc5391560c83cdcb029aa86bf983c2f66fd`, tag`v0.15.6`. It checked MSRV,
semver, security, exact source, packed source, downloaded registry witnesses
and GitHub release bytes before reporting success.

Public crate/index/GitHub archive SHA256:
`f0927b25a60092797fdb15bf850e60c89269dca07461c929901081ab1209f65d`.
Independent local verification compared345published source files and
Cargo.toml.orig against that exact Git commit, checked packaged VCS identity,
the crates.io index checksum, the GitHub tag and checksum asset. The private
receipt is retained as native0156-registry/verification-receipt-complete.json.

This HF change pins `mlx-native = "=0.15.6"` and that exact registry checksum.
No local path/ignored Cargo patch is used; unrelated lockfile dependency
selection is preserved. The native increment publishes opt-in live+pending
requested-root extent accounting and immutable host-identity caching, plus
the unchanged-bar M5 performance-gate warmup correction. These are not
whole-process physical-memory accounting or model-level speed proofs.

The new live authority is not implicitly enabled by this dependency pin.
HF device registration, workload policy, CPU/OS ledger and admission remain
separate work. Existing production defaults stay selected. The prior public
0.15.5model evidence remains bound to its original source and binaries.

Root validation against the clean registry dependency passed
`cargo check --locked --all-targets --all-features`, four actual dependency-pin
parser tests and the real tiny Qwen conversion/typed-dispatch sealing test.
The expected evidence version now follows the exact pin automatically, and
sealing accepted actual0.15.6 traces. Release-build, current composition and
real-model gates remain separately recorded; no all-four performance claim
or whole Q5 completion follows from adopting this public version.
