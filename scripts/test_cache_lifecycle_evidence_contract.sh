#!/usr/bin/env bash
set -euo pipefail

root_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)
packager="$root_dir/scripts/package_cache_lifecycle_evidence.sh"
tmp=$(mktemp -d "${TMPDIR:-/tmp}/hf2q-qualification-contract.XXXXXX")
cleanup() { rm -rf "$tmp"; }
trap cleanup EXIT

source_sha=aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa
version=0.1.21
candidate_run_id=123456
crate_sha=bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb
binary_sha=cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
input="$tmp/input"

sha256_file() { shasum -a 256 "$1" | awk '{print $1}'; }
expect_failure() {
  local qualification=$1
  local output=$2
  if "$packager" "$qualification" "$source_sha" "$version" \
    "$candidate_run_id" "$crate_sha" "$binary_sha" "$output" \
    >/dev/null 2>&1; then
    echo "qualification evidence packager accepted a mutant" >&2
    exit 1
  fi
}

mkdir -p "$input/deepseek/r2c-structured"
printf 'receipt bytes\n' >"$input/deepseek/r2c-structured/summary.json"
jq -nS --arg source_sha "$source_sha" --arg version "$version" \
  --arg candidate_run_id "$candidate_run_id" --arg crate_sha256 "$crate_sha" \
  --arg binary_sha256 "$binary_sha" '{
    kind:"hf2q.cache-lifecycle-release-manifest",schema_version:2,status:"pass",
    source_sha:$source_sha,version:$version,
    standalone_candidate_run_id:$candidate_run_id,
    crate_sha256:$crate_sha256,binary_sha256:$binary_sha256
  }' >"$input/manifest.json"
(cd "$input" && shasum -a 256 manifest.json >manifest.json.sha256)

"$packager" "$input" "$source_sha" "$version" "$candidate_run_id" \
  "$crate_sha" "$binary_sha" "$tmp/output-one" >"$tmp/output-one.env"
"$packager" "$input" "$source_sha" "$version" "$candidate_run_id" \
  "$crate_sha" "$binary_sha" "$tmp/output-two" >"$tmp/output-two.env"

archive="$tmp/output-one/hf2q-$version-qualification.tar.gz"
archive_checksum="$archive.sha256"
proof="$tmp/output-one/hf2q-$version-release-proof.json"
proof_checksum="$proof.sha256"
for file in "$archive" "$archive_checksum" "$proof" "$proof_checksum"; do
  [[ -s "$file" && ! -L "$file" ]]
done
cmp -s "$archive" "$tmp/output-two/$(basename "$archive")"
cmp -s "$proof" "$tmp/output-two/$(basename "$proof")"
read -r recorded_archive_sha recorded_archive_name extra <"$archive_checksum"
[[ -z "${extra:-}" && "$recorded_archive_name" == "$(basename "$archive")" ]]
[[ "$recorded_archive_sha" == "$(sha256_file "$archive")" ]]
read -r recorded_proof_sha recorded_proof_name extra <"$proof_checksum"
[[ -z "${extra:-}" && "$recorded_proof_name" == "$(basename "$proof")" ]]
[[ "$recorded_proof_sha" == "$(sha256_file "$proof")" ]]
jq -e --arg source_sha "$source_sha" --arg version "$version" \
  --arg candidate_run_id "$candidate_run_id" --arg crate_sha256 "$crate_sha" \
  --arg binary_sha256 "$binary_sha" --arg archive_sha256 "$recorded_archive_sha" '
    .kind == "hf2q.public-release-proof" and .schema_version == 1 and .status == "pass"
    and .source_sha == $source_sha and .version == $version
    and .standalone_candidate_run_id == $candidate_run_id
    and .crate_sha256 == $crate_sha256 and .binary_sha256 == $binary_sha256
    and .qualification.sha256 == $archive_sha256
  ' "$proof" >/dev/null

unpacked="$tmp/unpacked"
mkdir "$unpacked"
/usr/bin/tar -xzf "$archive" -C "$unpacked"
cmp -s "$input/manifest.json" \
  "$unpacked/hf2q-$version-qualification/manifest.json"
cmp -s "$input/deepseek/r2c-structured/summary.json" \
  "$unpacked/hf2q-$version-qualification/deepseek/r2c-structured/summary.json"

wrong_identity="$tmp/wrong-identity"
cp -R "$input" "$wrong_identity"
jq '.binary_sha256 = ("d" * 64)' "$wrong_identity/manifest.json" \
  >"$wrong_identity/manifest.json.tmp"
mv "$wrong_identity/manifest.json.tmp" "$wrong_identity/manifest.json"
(cd "$wrong_identity" && shasum -a 256 manifest.json >manifest.json.sha256)
expect_failure "$wrong_identity" "$tmp/wrong-identity-output"

linked="$tmp/linked"
cp -R "$input" "$linked"
ln -s "$input/manifest.json" "$linked/linked-manifest"
expect_failure "$linked" "$tmp/linked-output"

printf '%s\n' "cache lifecycle evidence contract: pass"
