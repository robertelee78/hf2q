#!/usr/bin/env bash
# Download and verify the exact packed-source and Developer-ID/notarization
# artifacts produced by one successful Standalone candidate workflow run.
set -euo pipefail

if [[ $# -ne 5 ]]; then
  echo "usage: $0 RUN_ID SOURCE_SHA VERSION OUTPUT_DIRECTORY GITHUB_ENV" >&2
  exit 2
fi

run_id=$1
source_sha=$2
version=$3
output_directory=$4
github_env=$5

fail() {
  echo "standalone candidate verification: $*" >&2
  exit 1
}

sha256_file() {
  shasum -a 256 "$1" | awk '{print $1}'
}

[[ "$run_id" =~ ^[1-9][0-9]*$ ]] || fail "workflow run ID is not canonical"
[[ "$source_sha" =~ ^[0-9a-f]{40}$ ]] || fail "source SHA is not canonical"
[[ "$version" =~ ^(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)$ ]] || \
  fail "version is not canonical stable SemVer"
[[ "$output_directory" == /* && ! -e "$output_directory" ]] || \
  fail "output directory must be a new absolute path"
[[ "$github_env" == /* && -f "$github_env" && ! -L "$github_env" ]] || \
  fail "GitHub environment file is not a regular absolute file"
[[ $(uname -s) == Darwin && $(uname -m) == arm64 ]] || \
  fail "verification requires an Apple-Silicon macOS runner"
: "${GH_TOKEN:?GH_TOKEN is required}"

run_json=$(gh run view "$run_id" \
  --json conclusion,event,headSha,workflowName,url)
[[ $(jq -r .workflowName <<<"$run_json") == "Standalone candidate" ]] || \
  fail "run is not a Standalone candidate workflow"
[[ $(jq -r .event <<<"$run_json") == workflow_dispatch ]] || \
  fail "candidate was not explicitly dispatched"
[[ $(jq -r .headSha <<<"$run_json") == "$source_sha" ]] || \
  fail "candidate source SHA differs from the release source"
[[ $(jq -r .conclusion <<<"$run_json") == success ]] || \
  fail "candidate workflow did not succeed"

mkdir -m 0700 "$output_directory"
build_root="$output_directory/build"
signed_root="$output_directory/signed"
gh run download "$run_id" \
  --name "standalone-candidate-build-$source_sha" --dir "$build_root"
gh run download "$run_id" \
  --name "standalone-candidate-signed-$source_sha" --dir "$signed_root"

[[ -z $(find "$build_root" "$signed_root" -type l -print -quit) ]] || \
  fail "candidate artifacts contain a symbolic link"
[[ $(find "$build_root" -mindepth 1 -maxdepth 1 -print | wc -l | tr -d ' ') == 4 ]] || \
  fail "build artifact inventory is not exact"
[[ $(find "$signed_root" -mindepth 1 -maxdepth 1 -print | wc -l | tr -d ' ') == 8 ]] || \
  fail "signed artifact inventory is not exact"

build_receipt="$build_root/build.json"
crate="$build_root/hf2q-${version}.crate"
unsigned="$build_root/hf2q-unsigned"
dependency_provenance="$build_root/dependency-provenance"
for required in "$build_receipt" "$crate" "$unsigned" \
  "$dependency_provenance/Cargo.lock" \
  "$dependency_provenance/cargo-metadata.json" \
  "$dependency_provenance/receipt.json"; do
  [[ -f "$required" && ! -L "$required" ]] || \
    fail "build artifact is missing $(basename "$required")"
done
jq -e \
  --arg source_sha "$source_sha" \
  --arg version "$version" '
    .kind == "hf2q.standalone-build-candidate"
    and .schema_version == 1
    and .source_sha == $source_sha
    and .version == $version
    and .target == "aarch64-apple-darwin"
    and .minimum_macos == "14.0"
    and (.crate_sha256 | test("^[0-9a-f]{64}$"))
    and (.unsigned_binary_sha256 | test("^[0-9a-f]{64}$"))
  ' "$build_receipt" >/dev/null || fail "build receipt is invalid"
crate_sha=$(jq -er .crate_sha256 "$build_receipt")
unsigned_sha=$(jq -er .unsigned_binary_sha256 "$build_receipt")
[[ $(sha256_file "$crate") == "$crate_sha" ]] || fail "crate digest differs from receipt"
[[ $(sha256_file "$unsigned") == "$unsigned_sha" ]] || \
  fail "unsigned binary digest differs from receipt"
chmod 0555 "$unsigned"
[[ $(/usr/bin/lipo -archs "$unsigned") == arm64 ]] || \
  fail "unsigned binary is not thin arm64"
[[ $(/usr/bin/vtool -show-build "$unsigned" | awk '$1 == "minos" {print $2}') == 14.0 ]] || \
  fail "unsigned binary minimum macOS is not 14.0"
[[ $("$unsigned" --version) == "hf2q $version" ]] || \
  fail "unsigned binary version differs from release version"

script_directory=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)
bash "$script_directory/verify_release_dependency_provenance.sh" verify \
  "$dependency_provenance" "$script_directory/../Cargo.lock"

for required in \
  hf2q-aarch64-apple-darwin \
  hf2q-aarch64-apple-darwin.sha256 \
  proof.json \
  notary-submission.json \
  notary-wait.json \
  notary-log.json \
  codesign.txt \
  notarization-check.txt; do
  [[ -f "$signed_root/$required" && ! -L "$signed_root/$required" ]] || \
    fail "signed artifact is missing $required"
done

proof="$signed_root/proof.json"
binary="$signed_root/hf2q-aarch64-apple-darwin"
binary_sha=$(jq -er .asset.sha256 "$proof")
binary_size=$(jq -er .asset.size "$proof")
team_id=$(jq -er .signing.team_id "$proof")
identifier=$(jq -er .signing.identifier "$proof")
jq -e \
  --arg source_sha "$source_sha" \
  --arg version "$version" \
  --arg unsigned_sha "$unsigned_sha" '
    .kind == "hf2q.standalone-apple-release-proof"
    and .schema_version == 1
    and .package == "hf2q"
    and .source_sha == $source_sha
    and .version == $version
    and .target == "aarch64-apple-darwin"
    and .input.unsigned_sha256 == $unsigned_sha
    and .asset.name == "hf2q-aarch64-apple-darwin"
    and (.asset.size | type) == "number"
    and .asset.size > 0
    and (.asset.sha256 | test("^[0-9a-f]{64}$"))
    and (.signing.team_id | test("^[A-Z0-9]{10}$"))
    and .signing.identifier == "us.hf2q.cli"
    and .signing.authority == "Developer ID Application"
    and .signing.hardened_runtime == true
    and .signing.secure_timestamp == true
    and .notarization.status == "Accepted"
    and .notarization.standalone_ticket_stapled == false
    and .verification.codesign == "accepted"
    and .verification.online_notarization == "accepted"
    and .verification.notary_ticket_cdhash_matches == true
  ' "$proof" >/dev/null || fail "signed proof is invalid"

for pair in \
  "notary-submission.json:.notarization.submission_sha256" \
  "notary-wait.json:.notarization.wait_sha256" \
  "notary-log.json:.notarization.log_sha256" \
  "codesign.txt:.verification.codesign_log_sha256" \
  "notarization-check.txt:.verification.online_notarization_log_sha256"; do
  file=${pair%%:*}
  selector=${pair#*:}
  [[ $(sha256_file "$signed_root/$file") == $(jq -er "$selector" "$proof") ]] || \
    fail "$file digest differs from proof"
done

chmod 0555 "$binary"
[[ $(sha256_file "$binary") == "$binary_sha" ]] || \
  fail "signed binary digest differs from proof"
[[ $(stat -f '%z' "$binary") == "$binary_size" ]] || \
  fail "signed binary size differs from proof"
read -r checksum_name checksum_file < \
  "$signed_root/hf2q-aarch64-apple-darwin.sha256"
[[ "$checksum_name" == "$binary_sha" && "$checksum_file" == hf2q-aarch64-apple-darwin ]] || \
  fail "signed binary checksum file is invalid"
[[ $(/usr/bin/lipo -archs "$binary") == arm64 ]] || \
  fail "signed binary is not thin arm64"
[[ $(/usr/bin/vtool -show-build "$binary" | awk '$1 == "minos" {print $2}') == 14.0 ]] || \
  fail "signed binary minimum macOS is not 14.0"
/usr/bin/codesign --verify --strict --all-architectures "$binary"
/usr/bin/codesign --verify --strict --all-architectures \
  --check-notarization --test-requirement '=notarized' "$binary"
codesign_info=$(/usr/bin/codesign --display --verbose=4 "$binary" 2>&1)
[[ $(grep -c '^CDHash=' <<<"$codesign_info") == 1 ]] || \
  fail "signed binary has ambiguous CDHash metadata"
cdhash=$(sed -n 's/^CDHash=//p' <<<"$codesign_info")
[[ "$cdhash" == $(jq -er .signing.cdhash "$proof") ]] || \
  fail "signed binary CDHash differs from proof"
jq -e --arg cdhash "$cdhash" '
  .status == "Accepted"
  and ((.issues // []) | length) == 0
  and any(.ticketContents[]?;
    .digestAlgorithm == "SHA-256" and .cdhash == $cdhash)
' "$signed_root/notary-log.json" >/dev/null || \
  fail "notary log does not bind the signed binary CDHash"
[[ $("$binary" --version) == "hf2q $version" ]] || \
  fail "signed binary version differs from release version"

{
  printf 'EXPECTED_CRATE_SHA256=%s\n' "$crate_sha"
  printf 'DEPENDENCY_PROVENANCE_DIR=%s\n' "$dependency_provenance"
  printf 'STANDALONE_RELEASE_DIR=%s\n' "$signed_root"
  printf 'STANDALONE_BINARY_SHA256=%s\n' "$binary_sha"
  printf 'STANDALONE_BINARY_SIZE=%s\n' "$binary_size"
  printf 'STANDALONE_TEAM_ID=%s\n' "$team_id"
  printf 'STANDALONE_IDENTIFIER=%s\n' "$identifier"
} >> "$github_env"

echo "standalone candidate verification: PASS"
