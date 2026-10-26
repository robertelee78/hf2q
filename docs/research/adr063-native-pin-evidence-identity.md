# Exact native pin identity for typed evidence

Candidate, 2026-10-26. Original stored-evidence failure is retained:
updating only a test literal to0.15.5 left the production sealer requiring0.15.4,
so actual published0.15.5 traces failed inconsistent-typed-evidence validation.
No author test/build/model execution or passing result is claimed here.

The canonical root `[dependencies] mlx-native = "=major.minor.patch"` in Cargo.toml
is now the SINGLE expected-version authority. Existing build_script reads it,
using shared build_native_pin parser and the already-pinned TOML1.1.4, and emits
HF2Q_MLX_NATIVE_VERSION via cargo:rustc-env. Cargo.toml is an explicit rerun input.
The Qwen encoded-dispatch sealer, official-source operator summary and actual
stored trace assertion all consume that build value with env!, rather than
separate version literals. A later exact0.15.6 pin therefore updates the expected
identity automatically; no version whitelist or approximate fallback is added.

Parser accepts both the source exact string pin and Cargo-normalized
`[dependencies.mlx-native] version = "=major.minor.patch"` table, including inline
table form. Full TOML parsing rejects duplicate/conflicting keys and malformed
syntax. Nonexact versions, target/aliased native versions, path/git/workspace or
patch/replacement sources, and missing native version fail closed. Source exact
identity must agree with actual Cargo.toml; Cargo.toml.orig is not a fallback. Historical
frozen records and generic fixture versions are unchanged. Lockfile/registry
provenance still requires normal locked builds and source/archive verification;
this expected version alone does not attest linked-source authenticity or defeat
an ignored localpathpatch. The same locked TOML1.1.4/parse-display-serde feature selection is added to
build-dependencies; no new package version or production runtime cost is added.
Root must verify locked/package builds; author has not run them.

Four pure controls compile the ACTUAL build parser through stored-evidence test
module: actual source/env equality, entire Cargo-generated historical0.1.24
manifest and futureexact normalizedtable; nonexact/malformed in bothforms;
duplicate/target/alias/override; comments/unrelated/source-inline-table forms.
The entire normalized fixture is retained unchanged from the public0.1.24
registry package, whose native0.15.1 is historical fixture data, never a live
producer whitelist. V1 source was archived before repair: it rejected normalized
Cargo tables and must not be shipped as package-safe. Root-focused filter
`native_pin_contract::tests` must select4, then existing stored-evidence actual
sealing gate must pass with exactpublisheddependency. All other typed schema,
request/layout/safety/dispatch/catalog checks remain unchanged. Source-only until
root validation; no broader Q5/model/speed claim.

Root validation: four actual parser controls passed, followed by the real tiny
conversion/typed-dispatch sealing test on Apple Silicon with public0.15.5.
`cargo check --locked --all-targets --all-features` passed. Cargo.lock did not
change because the identical TOML version/features were already locked. `cargo package --locked --offline` also passed, compiling the actual normalized
packaged manifest and build script. This verifies normal package compilation,
not the ignored package test targets. No whole-suite or speed claim follows.
