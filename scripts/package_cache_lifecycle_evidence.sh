#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 7 ]]; then
  echo "usage: $0 QUALIFICATION_ROOT SOURCE_SHA VERSION CANDIDATE_RUN_ID CRATE_SHA256 BINARY_SHA256 OUTPUT_DIR" >&2
  exit 2
fi

qualification_root=$1
source_sha=$2
version=$3
candidate_run_id=$4
crate_sha256=$5
binary_sha256=$6
output_dir=$7

fail() {
  echo "package cache lifecycle evidence: $*" >&2
  exit 1
}

sha256_file() { shasum -a 256 "$1" | awk '{print $1}'; }
file_bytes() {
  stat -f '%z' "$1" 2>/dev/null || stat -c '%s' "$1" 2>/dev/null
}

for command in awk cmp cp find gzip jq mktemp shasum stat tar touch tr wc; do
  command -v "$command" >/dev/null || fail "missing required command: $command"
done
[[ "$qualification_root" == /* && -d "$qualification_root" && ! -L "$qualification_root" ]] || \
  fail "qualification root must be an absolute, real directory"
[[ "$source_sha" =~ ^[0-9a-f]{40}$ ]] || fail "source SHA is not canonical"
[[ "$version" =~ ^(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)$ ]] || \
  fail "version is not canonical stable SemVer"
[[ "$candidate_run_id" =~ ^[1-9][0-9]*$ ]] || fail "candidate run ID is not canonical"
for digest in "$crate_sha256" "$binary_sha256"; do
  [[ "$digest" =~ ^[0-9a-f]{64}$ ]] || fail "artifact SHA-256 is not canonical"
done
[[ "$output_dir" == /* && ! -e "$output_dir" ]] || \
  fail "output directory must be a new absolute path"
[[ -z $(find "$qualification_root" -type l -print -quit) ]] || \
  fail "qualification evidence contains a symbolic link"

manifest="$qualification_root/manifest.json"
manifest_checksum="$qualification_root/manifest.json.sha256"
[[ -s "$manifest" && -s "$manifest_checksum" ]] || fail "manifest or checksum is missing"
[[ $(wc -l <"$manifest_checksum" | tr -d ' ') == 1 ]] || \
  fail "manifest checksum has extra records"
read -r manifest_sha manifest_name extra <"$manifest_checksum"
[[ -z "${extra:-}" && "$manifest_name" == manifest.json ]] || \
  fail "manifest checksum format is invalid"
[[ "$manifest_sha" == "$(sha256_file "$manifest")" ]] || fail "manifest checksum differs"
jq -e --arg source_sha "$source_sha" --arg version "$version" \
  --arg candidate_run_id "$candidate_run_id" --arg crate_sha256 "$crate_sha256" \
  --arg binary_sha256 "$binary_sha256" '
    .kind == "hf2q.cache-lifecycle-release-manifest"
    and .schema_version == 2 and .status == "pass"
    and .source_sha == $source_sha and .version == $version
    and .standalone_candidate_run_id == $candidate_run_id
    and .crate_sha256 == $crate_sha256 and .binary_sha256 == $binary_sha256
  ' "$manifest" >/dev/null || fail "manifest identity differs from the release candidate"

mkdir -m 0700 "$output_dir"
stage=$(mktemp -d "${TMPDIR:-/tmp}/hf2q-qualification-asset.XXXXXX")
verify_stage=$(mktemp -d "${TMPDIR:-/tmp}/hf2q-qualification-verify.XXXXXX")
cleanup() { rm -rf "$stage" "$verify_stage"; }
trap cleanup EXIT

bundle_name="hf2q-$version-qualification"
bundle_root="$stage/$bundle_name"
mkdir -m 0700 "$bundle_root"
cp -R "$qualification_root/." "$bundle_root/"
find "$bundle_root" -exec touch -h -t 200001010000 {} +
file_list="$stage/files.list"
(
  cd "$stage"
  find "$bundle_name" -type f -print | LC_ALL=C sort >"$file_list"
)
[[ -s "$file_list" ]] || fail "qualification evidence is empty"

archive="$output_dir/$bundle_name.tar.gz"
COPYFILE_DISABLE=1 /usr/bin/tar --format ustar --uid 0 --gid 0 \
  --uname root --gname wheel -cf - -C "$stage" -T "$file_list" | \
  gzip -n >"$archive"
gzip -t "$archive"
/usr/bin/tar -xzf "$archive" -C "$verify_stage"
[[ -z $(find "$verify_stage" -type l -print -quit) ]] || fail "archive contains a symbolic link"

input_count=$(find "$bundle_root" -type f | wc -l | tr -d ' ')
output_count=$(find "$verify_stage/$bundle_name" -type f | wc -l | tr -d ' ')
[[ "$input_count" == "$output_count" ]] || fail "archive file inventory differs"
while IFS= read -r relative; do
  cmp -s "$stage/$relative" "$verify_stage/$relative" || \
    fail "archive bytes differ: $relative"
done <"$file_list"

archive_sha=$(sha256_file "$archive")
archive_bytes=$(file_bytes "$archive")
archive_checksum="$archive.sha256"
printf '%s  %s\n' "$archive_sha" "$(basename "$archive")" >"$archive_checksum"

proof="$output_dir/hf2q-$version-release-proof.json"
jq -nS --arg source_sha "$source_sha" --arg version "$version" \
  --arg candidate_run_id "$candidate_run_id" --arg crate_sha256 "$crate_sha256" \
  --arg binary_sha256 "$binary_sha256" --arg manifest_sha256 "$manifest_sha" \
  --arg archive_name "$(basename "$archive")" --arg archive_sha256 "$archive_sha" \
  --argjson archive_bytes "$archive_bytes" '{
    kind:"hf2q.public-release-proof",schema_version:1,status:"pass",
    source_sha:$source_sha,version:$version,
    standalone_candidate_run_id:$candidate_run_id,
    crate_sha256:$crate_sha256,binary_sha256:$binary_sha256,
    qualification:{manifest_sha256:$manifest_sha256,asset:$archive_name,
      sha256:$archive_sha256,bytes:$archive_bytes}
  }' >"$proof"
proof_checksum="$proof.sha256"
printf '%s  %s\n' "$(sha256_file "$proof")" "$(basename "$proof")" >"$proof_checksum"

echo "QUALIFICATION_ARCHIVE=$archive"
echo "QUALIFICATION_ARCHIVE_SHA256=$archive_sha"
echo "QUALIFICATION_ARCHIVE_CHECKSUM=$archive_checksum"
echo "PUBLIC_RELEASE_PROOF=$proof"
echo "PUBLIC_RELEASE_PROOF_CHECKSUM=$proof_checksum"
