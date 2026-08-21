#!/usr/bin/env bash
# shellcheck disable=SC2016
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
VERIFIER="$ROOT_DIR/scripts/verify_release_dependency_provenance.sh"
CACHE_WORKFLOW="$ROOT_DIR/.github/workflows/cache-lifecycle.yml"
RELEASE_WORKFLOW="$ROOT_DIR/.github/workflows/release.yml"
RELEASE_GATE="$ROOT_DIR/scripts/run_agentic_cache_release_gate.sh"
scratch=$(mktemp -d "${TMPDIR:-/tmp}/hf2q-dependency-provenance.XXXXXX")
trap 'rm -rf "$scratch"' EXIT

fail() {
  echo "$*" >&2
  exit 1
}

expect_failure() {
  local description=$1
  local expected_stderr=$2
  local stderr_file
  shift 2
  failure_count=$((failure_count + 1))
  stderr_file="$scratch/expected-failure-${failure_count}.stderr"
  if "$@" >/dev/null 2>"$stderr_file"; then
    fail "$description"
  fi
  if ! grep -qF "$expected_stderr" "$stderr_file"; then
    sed 's/^/unexpected stderr: /' "$stderr_file" >&2
    fail "$description (wrong failure path; expected: $expected_stderr)"
  fi
}

failure_count=0

checkout="$scratch/checkout"
package_root="$scratch/packed/hf2q-0.1.7"
cargo_home="$scratch/cargo-home"
cargo_target="$scratch/cargo-target"
metadata="$scratch/cargo-metadata.json"
evidence="$scratch/evidence"
mkdir -p "$checkout" "$package_root" "$cargo_home" "$cargo_target"
checkout=$(cd "$checkout" && pwd -P)
package_root=$(cd "$package_root" && pwd -P)
cargo_home=$(cd "$cargo_home" && pwd -P)
cargo_target=$(cd "$cargo_target" && pwd -P)
crate_sha=bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb

cat > "$package_root/Cargo.lock" <<'LOCK'
version = 4

[[package]]
name = "hf2q"
version = "0.1.7"
dependencies = [
 "mlx-native",
]

[[package]]
name = "mlx-native"
version = "0.11.0"
source = "registry+https://github.com/rust-lang/crates.io-index"
checksum = "f9e25280262b20fd2894acc90229a7d2f695a1d818451473f95ef40125962cd8"
LOCK

jq -n --arg workspace_root "$package_root" '
  {
    packages: [
      {
        name:"hf2q",
        version:"0.1.7",
        dependencies:[{
          name:"mlx-native",
          source:"registry+https://github.com/rust-lang/crates.io-index",
          req:"=0.11.0"
        }]
      },
      {
        name:"mlx-native",
        version:"0.11.0",
        source:"registry+https://github.com/rust-lang/crates.io-index",
        dependencies:[]
      }
    ],
    workspace_root:$workspace_root
  }
' > "$metadata"

bash "$VERIFIER" capture "$package_root" "$metadata" "$evidence" \
  "$checkout" "$cargo_home" "$cargo_target" true true "$crate_sha"
bash "$VERIFIER" verify "$evidence" "$package_root/Cargo.lock"

wrong_checksum="$scratch/wrong-checksum"
cp -R "$evidence" "$wrong_checksum"
awk '{gsub(/f9e25280262b20fd2894acc90229a7d2f695a1d818451473f95ef40125962cd8/, "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa")}1' \
  "$wrong_checksum/Cargo.lock" > "$wrong_checksum/Cargo.lock.tmp"
mv "$wrong_checksum/Cargo.lock.tmp" "$wrong_checksum/Cargo.lock"
expect_failure "verifier accepted a substituted mlx-native checksum" \
  "Cargo.lock mlx-native checksum mismatch" \
  bash "$VERIFIER" verify "$wrong_checksum"

wrong_source="$scratch/wrong-source"
cp -R "$evidence" "$wrong_source"
jq '(.packages[] | select(.name == "mlx-native") | .source) = "git+https://example.invalid/mlx-native"' \
  "$wrong_source/cargo-metadata.json" > "$wrong_source/cargo-metadata.json.tmp"
mv "$wrong_source/cargo-metadata.json.tmp" "$wrong_source/cargo-metadata.json"
expect_failure "verifier accepted non-registry mlx-native metadata" \
  "cargo metadata dependency identity mismatch" \
  bash "$VERIFIER" verify "$wrong_source"

raw_hash_mismatch="$scratch/raw-hash-mismatch"
cp -R "$evidence" "$raw_hash_mismatch"
printf '\n' >> "$raw_hash_mismatch/cargo-metadata.json"
expect_failure "verifier accepted raw evidence whose receipt hash was stale" \
  "receipt identity or raw hashes mismatch" \
  bash "$VERIFIER" verify "$raw_hash_mismatch"

inside_checkout="$checkout/hf2q-0.1.7"
mkdir -p "$inside_checkout"
cp "$package_root/Cargo.lock" "$inside_checkout/Cargo.lock"
jq --arg workspace_root "$inside_checkout" '.workspace_root = $workspace_root' \
  "$metadata" > "$scratch/inside-metadata.json"
expect_failure "capture accepted a packed root inside checkout ancestry" \
  "packed package root is inside source checkout ancestry" \
  bash "$VERIFIER" capture "$inside_checkout" "$scratch/inside-metadata.json" \
    "$scratch/inside-evidence" "$checkout" "$cargo_home" "$cargo_target" \
    true true "$crate_sha"

local_config="$scratch/local-config/hf2q-0.1.7"
mkdir -p "$local_config/.cargo"
cp "$package_root/Cargo.lock" "$local_config/Cargo.lock"
printf '[build]\nrustflags = ["-C", "target-cpu=native"]\n' \
  > "$local_config/.cargo/config.toml"
jq --arg workspace_root "$local_config" '.workspace_root = $workspace_root' \
  "$metadata" > "$scratch/local-config-metadata.json"
expect_failure "capture accepted package-local Cargo configuration" \
  "packed package inherits forbidden Cargo configuration" \
  bash "$VERIFIER" capture "$local_config" "$scratch/local-config-metadata.json" \
    "$scratch/local-config-evidence" "$checkout" "$cargo_home" "$cargo_target" \
    true true "$crate_sha"

ancestor_config_root="$scratch/ancestor-config"
ancestor_package="$ancestor_config_root/nested/hf2q-0.1.7"
mkdir -p "$ancestor_config_root/.cargo" "$ancestor_package"
cp "$package_root/Cargo.lock" "$ancestor_package/Cargo.lock"
printf '[build]\nrustflags = ["-C", "target-cpu=native"]\n' \
  > "$ancestor_config_root/.cargo/config.toml"
jq --arg workspace_root "$ancestor_package" '.workspace_root = $workspace_root' \
  "$metadata" > "$scratch/ancestor-config-metadata.json"
expect_failure "capture accepted Cargo configuration inherited from a parent directory" \
  "packed package inherits forbidden Cargo configuration" \
  bash "$VERIFIER" capture "$ancestor_package" \
    "$scratch/ancestor-config-metadata.json" "$scratch/ancestor-config-evidence" \
    "$checkout" "$cargo_home" "$cargo_target" true true "$crate_sha"

expect_failure "capture accepted uncleared Rust flags or compiler wrappers" \
  "Rust build override environment was not cleared" \
  bash "$VERIFIER" capture "$package_root" "$metadata" \
    "$scratch/uncleared-rust-env-evidence" "$checkout" "$cargo_home" \
    "$cargo_target" true false "$crate_sha"

expect_failure "build environment check accepted a hostile Rust compiler flag" \
  "forbidden Rust build override environment variable is set: RUSTFLAGS" \
  env -i PATH="$PATH" CARGO_HOME="$cargo_home" \
    CARGO_TARGET_DIR="$cargo_target" RUSTFLAGS=-Ctarget-cpu=native \
    bash "$VERIFIER" check-build-env "$cargo_home" "$cargo_target"

expect_failure "build environment check accepted an ambient Rust toolchain" \
  "forbidden Rust build override environment variable is set: RUSTUP_TOOLCHAIN" \
  env -i PATH="$PATH" CARGO_HOME="$cargo_home" \
    CARGO_TARGET_DIR="$cargo_target" RUSTUP_TOOLCHAIN=nightly \
    bash "$VERIFIER" check-build-env "$cargo_home" "$cargo_target"

different_lock="$scratch/different-Cargo.lock"
cp "$package_root/Cargo.lock" "$different_lock"
printf '\n# release-side packed lock mutation\n' >> "$different_lock"
expect_failure "verifier accepted a different release-side packed Cargo.lock" \
  "downloaded evidence Cargo.lock differs from expected Cargo.lock" \
  bash "$VERIFIER" verify "$evidence" "$different_lock"

grep -qF 'packed_build_root=$(mktemp -d "/private/var/tmp/hf2q-packed-build.XXXXXX")' \
  "$CACHE_WORKFLOW" || fail "cache workflow does not allocate a config-isolated packed build root"
grep -qF '/usr/bin/tar -xzf "$crate" -C "$packed_build_root"' \
  "$CACHE_WORKFLOW" || fail "cache workflow does not unpack the sealed crate externally"
grep -qF 'cargo package --locked --no-verify' "$CACHE_WORKFLOW" || \
  fail "cache workflow performs ambient package verification before isolated build"
grep -qF 'test -z "$(find "$cargo_home" -mindepth 1 -print -quit)"' \
  "$CACHE_WORKFLOW" || fail "cache workflow does not prove Cargo home starts empty"
grep -qF 'cargo_env=(env)' "$CACHE_WORKFLOW" || \
  fail "cache workflow does not construct a controlled Cargo environment"
grep -qF 'RUSTUP_TOOLCHAIN|RUSTC|RUSTDOC|RUSTFLAGS|RUSTDOCFLAGS|' \
  "$CACHE_WORKFLOW" || fail "cache workflow does not clear Rust toolchain overrides"
grep -qF 'CARGO_BUILD_*|CARGO_PROFILE_*|CARGO_TARGET_*' "$CACHE_WORKFLOW" || \
  fail "cache workflow does not clear Cargo build/profile/target overrides"
grep -qF 'cargo_env+=(-u "$variable_name")' "$CACHE_WORKFLOW" || \
  fail "cache workflow does not unset discovered override variables"
grep -qF 'check-build-env "$cargo_home" "$cargo_target"' "$CACHE_WORKFLOW" || \
  fail "cache workflow does not verify the controlled Cargo environment"
grep -qF 'trusted_verifier="$checkout_root/scripts/verify_release_dependency_provenance.sh"' \
  "$CACHE_WORKFLOW" || fail "cache workflow does not use the trusted checkout verifier"
if grep -qF '"$package_root/scripts/verify_release_dependency_provenance.sh"' \
  "$CACHE_WORKFLOW"; then
  fail "cache workflow invokes the verifier from the package being attested"
fi
grep -qF 'check-build-root "$package_root" "$checkout_root"' "$CACHE_WORKFLOW" || \
  fail "cache workflow does not reject inherited Cargo configuration before resolution"
[[ $(grep -cF 'check-build-root' "$CACHE_WORKFLOW") -ge 3 ]] || \
  fail "cache workflow does not recheck build roots before and after Cargo build"
grep -qF 'bash "$trusted_verifier" capture' "$CACHE_WORKFLOW" || \
  fail "cache workflow does not capture dependency evidence from the packed crate"
grep -qF 'DEPENDENCY_PROVENANCE_DIR="$DEPENDENCY_PROVENANCE_DIR"' \
  "$CACHE_WORKFLOW" || fail "cache workflow does not pass dependency evidence to the gate"
if grep -qF 'package_root="$GITHUB_WORKSPACE/target/package/' "$CACHE_WORKFLOW"; then
  fail "cache workflow still builds the hardware binary under checkout ancestry"
fi

grep -qF 'verify_release_dependency_provenance.sh" verify' "$RELEASE_GATE" || \
  fail "release gate does not independently verify dependency evidence"
grep -qF 'dependency_provenance: $dependency_provenance[0]' "$RELEASE_GATE" || \
  fail "release manifest does not bind the dependency receipt"
grep -qF 'provenance:{dependency:$dependency_provenance_receipt_sha}' "$RELEASE_GATE" || \
  fail "release manifest does not bind the dependency receipt hash"
grep -qF 'verify_release_dependency_provenance.sh verify' "$RELEASE_WORKFLOW" || \
  fail "release workflow does not independently verify downloaded evidence"
grep -qF '"$DEPENDENCY_PROVENANCE_DIR" "$packed_lock"' "$RELEASE_WORKFLOW" || \
  fail "release workflow does not compare the newly packed Cargo.lock"
grep -qF 'release_package_stage=$(mktemp -d' "$RELEASE_WORKFLOW" || \
  fail "release workflow does not allocate a fresh extraction root"
grep -qF '/usr/bin/tar -xzf "$crate" -C "$release_package_stage"' \
  "$RELEASE_WORKFLOW" || fail "release workflow does not explicitly extract the crate"
grep -qF 'echo "RELEASE_PACKAGE_ROOT=$package_root" >> "$GITHUB_ENV"' \
  "$RELEASE_WORKFLOW" || fail "release workflow does not export the fresh package root"
grep -qF 'package_root=${RELEASE_PACKAGE_ROOT:?RELEASE_PACKAGE_ROOT is required}' \
  "$RELEASE_WORKFLOW" || fail "packed smoke tests do not consume the fresh package root"
if grep -qF 'packed_lock="target/package/hf2q-${EXPECTED_VERSION}/Cargo.lock"' \
  "$RELEASE_WORKFLOW"; then
  fail "release workflow still assumes cargo package leaves an unpacked directory"
fi
grep -qF '"$script_dir/../Cargo.lock"' "$RELEASE_GATE" || \
  fail "release gate lock comparison depends on the caller working directory"
grep -qF 'PACKED_BUILD_ROOT=%s' "$CACHE_WORKFLOW" || \
  fail "cache workflow does not export the external packed build root"
grep -qF '/private/var/tmp/hf2q-packed-build.*)' "$CACHE_WORKFLOW" || \
  fail "cache workflow cleanup lacks an exact packed-build root guard"

awk '
  /bash -n scripts\/qwen36_watchdog_validate\.sh/ { in_packed_bash_n = 1 }
  in_packed_bash_n && /scripts\/verify_release_dependency_provenance\.sh/ {
    saw_verifier = 1
  }
  in_packed_bash_n && /scripts\/test_release_dependency_provenance_contract\.sh/ {
    saw_contract = 1
  }
  in_packed_bash_n && /scripts\/run_agentic_cache_release_gate\.sh/ {
    exit(saw_verifier && saw_contract ? 0 : 1)
  }
  END {
    if (!in_packed_bash_n || !saw_verifier || !saw_contract) exit 1
  }
' "$RELEASE_WORKFLOW" || \
  fail "release packed-source bash syntax gate omits dependency provenance scripts"
grep -qF 'bash scripts/test_release_dependency_provenance_contract.sh' \
  "$RELEASE_WORKFLOW" || \
  fail "release packed-source gate does not execute the dependency provenance contract"

operational_version_files=(
  "$RELEASE_WORKFLOW"
  "$RELEASE_GATE"
  "$ROOT_DIR/scripts/run_deepseek4_decode_cohort_gate.sh"
  "$ROOT_DIR/scripts/verify_deepseek4_decode_cohort_receipt.sh"
  "$ROOT_DIR/scripts/verify_deepseek4_cooperative_prefill_receipt.sh"
  "$ROOT_DIR/scripts/test_deepseek4_decode_cohort_receipt_contract.sh"
  "$ROOT_DIR/scripts/test_deepseek4_cooperative_prefill_receipt_contract.sh"
)
for operational_version_file in "${operational_version_files[@]}"; do
  if grep -qF '0.10.12' "$operational_version_file"; then
    fail "operational receipt or release gate still claims mlx-native 0.10.12: $operational_version_file"
  fi
done

echo "release dependency provenance contract: PASS"
