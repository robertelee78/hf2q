# Native 0.15.3 registry adoption

Date: 2026-10-25. This source change pins `mlx-native = "=0.15.3"` from
crates.io. It is a downstream dependency adoption, not a new hf2q release.
The hf2q crate version remains 0.1.27; historical published stock and prior
0.15.1/0.15.2 records retain their original identities.

## Verified public package

- Registry archive SHA-256:
  `edd1820a5e05694d8567b23159bcec950e117f863f7438eda8d6c143b615d54c`.
- Embedded VCS commit: `f92ac043435cc6a14ac1733c63d52526dcae8714`.
- [Registry package](https://crates.io/crates/mlx-native/0.15.3).
- [Native release](https://github.com/robertelee78/mlx-native/releases/tag/v0.15.3).
- [Successful exact-source release workflow 40142974346](https://github.com/robertelee78/mlx-native/actions/runs/40142974346).

Independent retained-evidence review verified the registry archive, GitHub
asset, checksum sidecar, tag and release workflow against that commit. The
entire archive and all 640 package members equal the trusted locked Cargo 1.96
local package; 637 nongenerated files also match Git source, with the three
Cargo-generated members separately verified. The public archive was compiled
in an isolated Cargo home/target and outside-home working directory using
Cargo/rustc 1.96.0, without local fixture overlays. Both selected debug and
release `--no-run` builds passed; six compiler-emitted test binaries were
archived and hashed. Cargo's automatic non-test helper binaries were recorded
but never executed. The initial debug artifact-parser assertion and early
recovery temp-directory lookup stop were wrapper failures; the successful
original debug build was preserved without recompilation.

## Public primitive controls

The downloaded package passed exactly 19 declared tests in 12 fresh control
processes on Apple M5 Max, arm64, macOS 27.2 build 26B5112a and SDK 27.0.
All controls exited successfully with no failed, ignored, unavailable or
pending tests, and bounded supervisors confirmed cleanup/reap. Evidence covers:

- Eight mapped-backing lifetime controls, including independent raw/recorded
  owners and nil/unwind cleanup.
- Five mandatory runtime-source selected-shader controls: prefill/training
  compilation, Qwen3.8/Qwen3.6 causal layouts and training output/LSE parity.
- The chunk arena's 13-root constructor/parameter control and a separate
  convenience-operation GPU FLA regression; this is not `_with_arena` GPU proof.
- Separate legacy-growth CPU and ordered-GPU controls with historical quota,
  lineage and CPU/standalone parity assertions.
- The bounded query-tiled indexer's three tiny causal/tie/offset cases,
  captured replay, streamed CPU and untiled parity, guards and zero-encode
  undersized-workspace rejection.
- One predeclared release indexer operator timing run, described below.

The strict receipt parser also passed 15 pure parser tests. These test parser
rejection logic, not GPU or model behavior. Exact public source, binary, log,
process, observer and compiler provenance are retained in the private ADR063
evidence archive. No raw host paths, model weights or captures are committed.

The one public operator run used Q2048/K63488/start251904, nine requested Metal
roots totaling 1,231,029,376 bytes, and 1,571,840 bytes of bounded numeric host
storage. It performed two warmups per arm and five ABBA blocks: 20 timings,
10 per arm. Wall medians were untiled 0.261680292 s and tiled 0.246486604 s,
a tiled/untiled ratio of 0.9419379736858442 (**5.8062026314% lower**).
All 14 passive thermal/power endpoints were nominal/low-power-off; this is
endpoint observation, not continuous thermal proof. Four actual warmed
pipeline identities were `PrecompiledMetallib`, with embedded SHA-256
`1535691acfc89a29734171c286a4a61ddfa92f971d47825cf6adbd7429782e5b`.
The run verified 21 independent CPU rows, exact full paired outputs and guards.
It is one host/operator experiment, not repeated-run statistical evidence or
a full-model speedup. The earlier candidate's 5.4175% result remains separate
historical evidence and is not overwritten by the public result.

## Downstream scope and remaining validation

This pin makes the published primitives available; it does not select the
additive indexer tiling, bounded pool, growth authority or any other new native
API in hf2q. No context, slot, quantum, speculation, model policy, kernel call
or sampling change is made. Existing mapped-backing ownership guarantees remain
in force. Logical workspace savings do not establish physical reclamation,
whole-model fit or complete driver/host overhead bounds.

Live version/checksum validators, newly produced release receipts, execution
trace assertions and source-teacher labels move together to the linked 0.15.3
identity. Historical manifests/receipts and synthetic negative fixtures stay
unchanged. A future campaign must freeze its actual new control and candidate
identities at campaign start rather than relabel earlier stock or F1 evidence.

Root downstream validation passed fresh isolated registry resolution: the cached
archive checksum/VCS and all 640 extracted files match the verified public
package. Only Cargo's generated `.cargo-ok` marker is additional. The lock edit
changes only the native version/checksum, with no path/git override.

Locked all-target/all-feature check passed (26.32s), the locked release build
passed (79.02s), and the test binary compiled (71.55s). Three exact tests passed:
runtime-trace capability, tensor-lineage rejection and the authoritative stored-
evidence conversion loop. The last control positively witnessed both generated
tiny Q8 and Q4_K_M fixtures' actual GPU branches, each reporting linked native
0.15.3 and 30 trace observations. Missing device/branch execution cannot qualify
that control. The initial marker borrow-after-seal compile error was corrected
and reviewed before these passing checks; it executed no tests or GPU work.

Release dependency provenance, DeepSeek decode/cooperative receipt contract
suites and all 29 generic identity tests also passed. Independent retained-
evidence review verified the source, binary, raw logs, process receipts and
positive marker counts. Final CI remains pending. The four-family real-model
quality/tool/SSE, speed and admission campaign remains open; this pin does not
qualify full Q5.
