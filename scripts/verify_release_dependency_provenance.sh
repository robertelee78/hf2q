#!/usr/bin/env bash
set -euo pipefail

readonly EXPECTED_PACKAGE_NAME=hf2q
SCRIPT_DIRECTORY=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)
readonly SCRIPT_DIRECTORY
readonly PACKAGE_MANIFEST="$SCRIPT_DIRECTORY/../Cargo.toml"
EXPECTED_PACKAGE_VERSION=$(sed -n 's/^version = "\([^"]*\)"/\1/p' \
  "$PACKAGE_MANIFEST" | head -1)
readonly EXPECTED_PACKAGE_VERSION
[[ "$EXPECTED_PACKAGE_VERSION" =~ ^(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)$ ]] || {
  echo "release dependency provenance: package version is not canonical stable SemVer" >&2
  exit 1
}
readonly EXPECTED_DEPENDENCY_NAME=mlx-native
readonly EXPECTED_DEPENDENCY_VERSION=0.15.1
readonly EXPECTED_DEPENDENCY_REQUIREMENT='=0.15.1'
readonly EXPECTED_DEPENDENCY_SOURCE='registry+https://github.com/rust-lang/crates.io-index'
readonly EXPECTED_DEPENDENCY_CHECKSUM=76ce4c8d5773c72554a98020aadd330e566792dd27f843451ac5dd567bb6b5dd

fail() {
  echo "release dependency provenance: $*" >&2
  exit 1
}

sha256_file() {
  shasum -a 256 "$1" | awk '{print $1}'
}

physical_dir() {
  (cd "$1" && pwd -P)
}

is_within() {
  local child=$1
  local parent=$2
  [[ "$child" == "$parent" || "$child" == "$parent/"* ]]
}

reject_ancestor_cargo_config() {
  local current=$1
  while :; do
    [[ ! -e "$current/.cargo/config" && ! -e "$current/.cargo/config.toml" ]] || \
      fail "packed package inherits forbidden Cargo configuration from: $current/.cargo"
    [[ "$current" == / ]] && break
    current=${current%/*}
    [[ -n "$current" ]] || current=/
  done
}

is_rust_build_override_name() {
  case "$1" in
    RUSTUP_TOOLCHAIN | RUSTC | RUSTDOC | RUSTFLAGS | RUSTDOCFLAGS | \
      CARGO_ENCODED_RUSTFLAGS | CARGO_ENCODED_RUSTDOCFLAGS | \
      RUSTC_* | RUSTDOC_* | CARGO_BUILD_* | CARGO_PROFILE_* | CARGO_TARGET_*)
      return 0
      ;;
    *)
      return 1
      ;;
  esac
}

verify_rust_build_environment() {
  local expected_cargo_home=${1:?expected Cargo home is required}
  local expected_cargo_target=${2:?expected Cargo target is required}
  local expected_home_physical expected_target_physical variable_name

  [[ -d "$expected_cargo_home" && -d "$expected_cargo_target" ]] || \
    fail "expected Cargo build roots are missing"
  expected_home_physical=$(physical_dir "$expected_cargo_home")
  expected_target_physical=$(physical_dir "$expected_cargo_target")
  [[ -n ${CARGO_HOME:-} && -d "$CARGO_HOME" \
    && $(physical_dir "$CARGO_HOME") == "$expected_home_physical" ]] || \
    fail "CARGO_HOME does not select the isolated Cargo home"
  [[ -n ${CARGO_TARGET_DIR:-} && -d "$CARGO_TARGET_DIR" \
    && $(physical_dir "$CARGO_TARGET_DIR") == "$expected_target_physical" ]] || \
    fail "CARGO_TARGET_DIR does not select the isolated Cargo target"

  while IFS='=' read -r variable_name _; do
    [[ "$variable_name" == CARGO_TARGET_DIR ]] && continue
    if is_rust_build_override_name "$variable_name"; then
      fail "forbidden Rust build override environment variable is set: $variable_name"
    fi
  done < <(env)
}

verify_package_root() {
  local package_root=${1:?packed package root is required}
  local package_physical

  [[ -d "$package_root" ]] || fail "packed package root is missing"
  package_physical=$(physical_dir "$package_root")
  reject_ancestor_cargo_config "$package_physical"
}

verify_build_roots() {
  local package_root=${1:?packed package root is required}
  local checkout_root=${2:?source checkout root is required}
  local cargo_home=${3:?isolated Cargo home is required}
  local cargo_target=${4:?isolated Cargo target is required}
  local package_physical checkout_physical cargo_home_physical cargo_target_physical

  [[ -d "$package_root" && -d "$checkout_root" \
    && -d "$cargo_home" && -d "$cargo_target" ]] || fail "build roots are missing"
  package_physical=$(physical_dir "$package_root")
  checkout_physical=$(physical_dir "$checkout_root")
  cargo_home_physical=$(physical_dir "$cargo_home")
  cargo_target_physical=$(physical_dir "$cargo_target")
  reject_ancestor_cargo_config "$package_physical"
  is_within "$package_physical" "$checkout_physical" && \
    fail "packed package root is inside source checkout ancestry"
  is_within "$cargo_home_physical" "$checkout_physical" && \
    fail "isolated Cargo home is inside source checkout ancestry"
  is_within "$cargo_target_physical" "$checkout_physical" && \
    fail "isolated Cargo target is inside source checkout ancestry"
  [[ "$cargo_home_physical" != "$cargo_target_physical" ]] || \
    fail "isolated Cargo home and target must be distinct"
}

read_mlx_lock_identity() {
  local lock=$1
  awk '
    function unquote(value) {
      sub(/^[^"]*"/, "", value)
      sub(/".*/, "", value)
      return value
    }
    function finish_package() {
      if (in_package && package_name == "mlx-native") {
        matches++
        found_version = package_version
        found_source = package_source
        found_checksum = package_checksum
      }
    }
    $0 == "[[package]]" {
      finish_package()
      in_package = 1
      package_name = ""
      package_version = ""
      package_source = ""
      package_checksum = ""
      next
    }
    in_package && /^name = "/ { package_name = unquote($0); next }
    in_package && /^version = "/ { package_version = unquote($0); next }
    in_package && /^source = "/ { package_source = unquote($0); next }
    in_package && /^checksum = "/ { package_checksum = unquote($0); next }
    END {
      finish_package()
      printf "%d\t%s\t%s\t%s\n", matches, found_version, found_source, found_checksum
    }
  ' "$lock"
}

verify_raw_semantics() {
  local lock=$1
  local metadata=$2
  local matches version source checksum

  IFS=$'\t' read -r matches version source checksum < <(read_mlx_lock_identity "$lock")
  [[ "$matches" == 1 ]] || fail "Cargo.lock must contain exactly one mlx-native package"
  [[ "$version" == "$EXPECTED_DEPENDENCY_VERSION" ]] || \
    fail "Cargo.lock mlx-native version mismatch: $version"
  [[ "$source" == "$EXPECTED_DEPENDENCY_SOURCE" ]] || \
    fail "Cargo.lock mlx-native source mismatch: $source"
  [[ "$checksum" == "$EXPECTED_DEPENDENCY_CHECKSUM" ]] || \
    fail "Cargo.lock mlx-native checksum mismatch: $checksum"

  jq -e \
    --arg package_name "$EXPECTED_PACKAGE_NAME" \
    --arg package_version "$EXPECTED_PACKAGE_VERSION" \
    --arg dependency_name "$EXPECTED_DEPENDENCY_NAME" \
    --arg dependency_version "$EXPECTED_DEPENDENCY_VERSION" \
    --arg dependency_requirement "$EXPECTED_DEPENDENCY_REQUIREMENT" \
    --arg dependency_source "$EXPECTED_DEPENDENCY_SOURCE" '
      ([.packages[] | select(.name == $package_name)] | length) == 1
      and ([.packages[]
        | select(.name == $package_name and .version == $package_version)
        | .dependencies[]
        | select(.name == $dependency_name
          and .req == $dependency_requirement
          and .source == $dependency_source)] | length) == 1
      and ([.packages[]
        | select(.name == $dependency_name
          and .version == $dependency_version
          and .source == $dependency_source)] | length) == 1
      and (.workspace_root | type == "string" and length > 1)
    ' "$metadata" >/dev/null || fail "cargo metadata dependency identity mismatch"
}

capture_evidence() {
  local package_root=${1:?packed package root is required}
  local metadata=${2:?cargo metadata JSON is required}
  local evidence_dir=${3:?evidence directory is required}
  local checkout_root=${4:?source checkout root is required}
  local cargo_home=${5:?isolated Cargo home is required}
  local cargo_target=${6:?isolated Cargo target is required}
  local cargo_home_was_fresh=${7:?Cargo home freshness assertion is required}
  local rust_build_override_env_was_cleared=${8:?Rust build override environment assertion is required}
  local crate_sha256=${9:?packed crate SHA-256 is required}
  local package_physical checkout_physical cargo_home_physical cargo_target_physical metadata_root
  local lock_out metadata_out receipt_out lock_sha metadata_sha

  [[ "$cargo_home_was_fresh" == true ]] || fail "Cargo home was not fresh before resolution"
  [[ "$rust_build_override_env_was_cleared" == true ]] || \
    fail "Rust build override environment was not cleared"
  [[ "$crate_sha256" =~ ^[0-9a-f]{64}$ ]] || fail "packed crate SHA-256 is malformed"
  [[ -d "$package_root" && -f "$package_root/Cargo.lock" ]] || \
    fail "packed package root is missing Cargo.lock"
  [[ -s "$metadata" && ! -L "$metadata" ]] || fail "cargo metadata evidence is missing"

  verify_build_roots "$package_root" "$checkout_root" "$cargo_home" "$cargo_target"
  package_physical=$(physical_dir "$package_root")
  checkout_physical=$(physical_dir "$checkout_root")
  cargo_home_physical=$(physical_dir "$cargo_home")
  cargo_target_physical=$(physical_dir "$cargo_target")

  verify_raw_semantics "$package_physical/Cargo.lock" "$metadata"
  metadata_root=$(jq -er '.workspace_root' "$metadata")
  [[ "$metadata_root" == "$package_physical" ]] || \
    fail "cargo metadata workspace_root is not the packed package root"

  mkdir -p "$evidence_dir"
  lock_out="$evidence_dir/Cargo.lock"
  metadata_out="$evidence_dir/cargo-metadata.json"
  receipt_out="$evidence_dir/receipt.json"
  for output in "$lock_out" "$metadata_out" "$receipt_out"; do
    [[ ! -e "$output" ]] || fail "refusing to overwrite evidence: $output"
  done
  cp "$package_physical/Cargo.lock" "$lock_out"
  cp "$metadata" "$metadata_out"
  lock_sha=$(sha256_file "$lock_out")
  metadata_sha=$(sha256_file "$metadata_out")

  jq -n \
    --arg package_name "$EXPECTED_PACKAGE_NAME" \
    --arg package_version "$EXPECTED_PACKAGE_VERSION" \
    --arg crate_sha256 "$crate_sha256" \
    --arg checkout_root "$checkout_physical" \
    --arg package_root "$package_physical" \
    --arg cargo_home "$cargo_home_physical" \
    --arg cargo_target "$cargo_target_physical" \
    --arg dependency_name "$EXPECTED_DEPENDENCY_NAME" \
    --arg dependency_version "$EXPECTED_DEPENDENCY_VERSION" \
    --arg dependency_requirement "$EXPECTED_DEPENDENCY_REQUIREMENT" \
    --arg dependency_source "$EXPECTED_DEPENDENCY_SOURCE" \
    --arg dependency_checksum "$EXPECTED_DEPENDENCY_CHECKSUM" \
    --arg cargo_lock_sha256 "$lock_sha" \
    --arg cargo_metadata_sha256 "$metadata_sha" '
      {
        schema_version: 1,
        status: "pass",
        package: {
          name: $package_name,
          version: $package_version,
          source: "packed-crate",
          crate_sha256: $crate_sha256
        },
        build: {
          checkout_root: $checkout_root,
          package_root: $package_root,
          cargo_home: $cargo_home,
          cargo_target: $cargo_target,
          workspace_ancestry_disjoint: true,
          cargo_target_checkout_disjoint: true,
          cargo_home_was_fresh_before_resolution: true,
          rust_build_override_env_cleared: true,
          rust_build_override_env_policy: [
            "RUSTUP_TOOLCHAIN", "RUSTC", "RUSTDOC", "RUSTFLAGS", "RUSTDOCFLAGS",
            "CARGO_ENCODED_RUSTFLAGS", "CARGO_ENCODED_RUSTDOCFLAGS",
            "RUSTC_*", "RUSTDOC_*", "CARGO_BUILD_*", "CARGO_PROFILE_*",
            "CARGO_TARGET_* except controlled CARGO_TARGET_DIR"
          ]
        },
        dependency: {
          name: $dependency_name,
          version: $dependency_version,
          requirement: $dependency_requirement,
          source: $dependency_source,
          checksum: $dependency_checksum
        },
        raw: {
          cargo_lock: {path: "Cargo.lock", sha256: $cargo_lock_sha256},
          cargo_metadata: {path: "cargo-metadata.json", sha256: $cargo_metadata_sha256}
        }
      }
    ' > "$receipt_out"

  verify_evidence "$evidence_dir" "$package_physical/Cargo.lock"
}

verify_evidence() {
  local evidence_dir=${1:?evidence directory is required}
  local expected_packed_lock=${2:-}
  local lock="$evidence_dir/Cargo.lock"
  local metadata="$evidence_dir/cargo-metadata.json"
  local receipt="$evidence_dir/receipt.json"
  local lock_sha metadata_sha package_root checkout_root cargo_home cargo_target metadata_root

  for input in "$lock" "$metadata" "$receipt"; do
    [[ -s "$input" && ! -L "$input" ]] || fail "missing or linked evidence file: $input"
  done
  verify_raw_semantics "$lock" "$metadata"
  lock_sha=$(sha256_file "$lock")
  metadata_sha=$(sha256_file "$metadata")

  jq -e \
    --arg package_name "$EXPECTED_PACKAGE_NAME" \
    --arg package_version "$EXPECTED_PACKAGE_VERSION" \
    --arg dependency_name "$EXPECTED_DEPENDENCY_NAME" \
    --arg dependency_version "$EXPECTED_DEPENDENCY_VERSION" \
    --arg dependency_requirement "$EXPECTED_DEPENDENCY_REQUIREMENT" \
    --arg dependency_source "$EXPECTED_DEPENDENCY_SOURCE" \
    --arg dependency_checksum "$EXPECTED_DEPENDENCY_CHECKSUM" \
    --arg cargo_lock_sha256 "$lock_sha" \
    --arg cargo_metadata_sha256 "$metadata_sha" '
      .schema_version == 1
      and .status == "pass"
      and .package.name == $package_name
      and .package.version == $package_version
      and .package.source == "packed-crate"
      and (.package.crate_sha256 | test("^[0-9a-f]{64}$"))
      and .build.workspace_ancestry_disjoint == true
      and .build.cargo_target_checkout_disjoint == true
      and .build.cargo_home_was_fresh_before_resolution == true
      and .build.rust_build_override_env_cleared == true
      and .build.rust_build_override_env_policy == [
        "RUSTUP_TOOLCHAIN", "RUSTC", "RUSTDOC", "RUSTFLAGS", "RUSTDOCFLAGS",
        "CARGO_ENCODED_RUSTFLAGS", "CARGO_ENCODED_RUSTDOCFLAGS",
        "RUSTC_*", "RUSTDOC_*", "CARGO_BUILD_*", "CARGO_PROFILE_*",
        "CARGO_TARGET_* except controlled CARGO_TARGET_DIR"
      ]
      and (.build.checkout_root | type == "string" and length > 1)
      and (.build.package_root | type == "string" and length > 1)
      and (.build.cargo_home | type == "string" and length > 1)
      and (.build.cargo_target | type == "string" and length > 1)
      and .dependency == {
        name:$dependency_name,
        version:$dependency_version,
        requirement:$dependency_requirement,
        source:$dependency_source,
        checksum:$dependency_checksum
      }
      and .raw.cargo_lock == {path:"Cargo.lock", sha256:$cargo_lock_sha256}
      and .raw.cargo_metadata == {path:"cargo-metadata.json", sha256:$cargo_metadata_sha256}
    ' "$receipt" >/dev/null || fail "receipt identity or raw hashes mismatch"

  package_root=$(jq -er '.build.package_root' "$receipt")
  checkout_root=$(jq -er '.build.checkout_root' "$receipt")
  cargo_home=$(jq -er '.build.cargo_home' "$receipt")
  cargo_target=$(jq -er '.build.cargo_target' "$receipt")
  metadata_root=$(jq -er '.workspace_root' "$metadata")
  [[ "$metadata_root" == "$package_root" ]] || fail "metadata root does not match receipt"
  is_within "$package_root" "$checkout_root" && fail "receipt package root is inside checkout"
  is_within "$cargo_home" "$checkout_root" && fail "receipt Cargo home is inside checkout"
  is_within "$cargo_target" "$checkout_root" && fail "receipt Cargo target is inside checkout"
  [[ "$cargo_home" != "$cargo_target" ]] || fail "receipt Cargo home and target are identical"

  if [[ -n "$expected_packed_lock" ]]; then
    [[ -s "$expected_packed_lock" && ! -L "$expected_packed_lock" ]] || \
      fail "expected packed Cargo.lock is missing"
    cmp -s "$lock" "$expected_packed_lock" || \
      fail "downloaded evidence Cargo.lock differs from expected Cargo.lock"
  fi
}

mode=${1:-}
case "$mode" in
  capture)
    shift
    capture_evidence "$@"
    ;;
  check-build-root)
    shift
    verify_build_roots "$@"
    ;;
  check-build-env)
    shift
    verify_rust_build_environment "$@"
    ;;
  check-package-root)
    shift
    verify_package_root "$@"
    ;;
  verify)
    shift
    verify_evidence "$@"
    ;;
  *)
    fail "usage: $0 check-package-root PACKAGE_ROOT | check-build-root PACKAGE_ROOT CHECKOUT_ROOT CARGO_HOME CARGO_TARGET | check-build-env CARGO_HOME CARGO_TARGET | capture PACKAGE_ROOT METADATA_JSON EVIDENCE_DIR CHECKOUT_ROOT CARGO_HOME CARGO_TARGET true true CRATE_SHA256 | verify EVIDENCE_DIR [PACKED_CARGO_LOCK]"
    ;;
esac
