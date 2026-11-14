#!/usr/bin/env bash
set -euo pipefail

root_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)
packager="$root_dir/scripts/package_cache_lifecycle_evidence.sh"
tmp=$(mktemp -d "${TMPDIR:-/tmp}/hf2q-qualification-contract.XXXXXX")
cleanup() { rm -rf "$tmp"; }
trap cleanup EXIT

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

# A passing archive starts from the same complete immutable evidence fixture
# used by the model-free ledger verifier tests, never a fabricated manifest.
python3 -B - "$root_dir/scripts" "$input" <<'PY'
import sys
from pathlib import Path

sys.path.insert(0, sys.argv[1])
from test_cache_release_orchestrator import make_evidence

make_evidence(Path(sys.argv[2]))
PY
source_sha=$(jq -er .source_sha "$input/manifest.json")
version=$(jq -er .version "$input/manifest.json")
candidate_run_id=$(jq -er .standalone_candidate_run_id "$input/manifest.json")
crate_sha=$(jq -er .crate_sha256 "$input/manifest.json")
binary_sha=$(jq -er .binary_sha256 "$input/manifest.json")

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
diff -r "$input" "$unpacked/hf2q-$version-qualification"

wrong_identity="$tmp/wrong-identity"
cp -R "$input" "$wrong_identity"
jq '.binary_sha256 = ("d" * 64)' "$wrong_identity/manifest.json" \
  >"$wrong_identity/manifest.json.tmp"
mv "$wrong_identity/manifest.json.tmp" "$wrong_identity/manifest.json"
(cd "$wrong_identity" && shasum -a 256 manifest.json >manifest.json.sha256)
expect_failure "$wrong_identity" "$tmp/wrong-identity-output"

# A checksum cannot bless an old manifest or a fabricated ledger binding.
for mutation in legacy missing-ledger empty-ledger; do
  mutant="$tmp/$mutation"
  cp -R "$input" "$mutant"
  case "$mutation" in
    legacy) filter='.schema_version = 2 | del(.phase_ledger)' ;;
    missing-ledger) filter='del(.phase_ledger)' ;;
    empty-ledger) filter='.phase_ledger = {}' ;;
  esac
  jq "$filter" "$mutant/manifest.json" >"$mutant/manifest.json.tmp"
  mv "$mutant/manifest.json.tmp" "$mutant/manifest.json"
  (cd "$mutant" && shasum -a 256 manifest.json >manifest.json.sha256)
  expect_failure "$mutant" "$tmp/$mutation-output"
done

missing_evidence="$tmp/missing-evidence"
cp -R "$input" "$missing_evidence"
rm -rf "$missing_evidence/.cache-release"
expect_failure "$missing_evidence" "$tmp/missing-evidence-output"

tampered_evidence="$tmp/tampered-evidence"
cp -R "$input" "$tampered_evidence"
plan_sha=$(jq -er .phase_ledger.plan_sha256 "$input/manifest.json")
evidence_file=""
while IFS= read -r candidate; do
  if [[ "$(sha256_file "$candidate")" == "$plan_sha" ]]; then
    evidence_file=$candidate
    break
  fi
done < <(find "$tampered_evidence/.cache-release" -type f -print)
[[ -n "$evidence_file" ]]
printf '\ntamper\n' >>"$evidence_file"
expect_failure "$tampered_evidence" "$tmp/tampered-evidence-output"

linked="$tmp/linked"
cp -R "$input" "$linked"
ln -s "$input/manifest.json" "$linked/linked-manifest"
expect_failure "$linked" "$tmp/linked-output"

printf '%s\n' "cache lifecycle evidence contract: pass"
