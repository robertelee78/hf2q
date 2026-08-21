#!/usr/bin/env bash
# shellcheck disable=SC1003,SC2016
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
SEAL_SCRIPT="$ROOT_DIR/scripts/seal_release_binary.sh"
STANDALONE_WORKFLOW="$ROOT_DIR/.github/workflows/standalone-candidate.yml"
VERIFY_SCRIPT="$ROOT_DIR/scripts/verify_standalone_candidate.sh"
RELEASE_WORKFLOW="$ROOT_DIR/.github/workflows/release.yml"
scratch=$(mktemp -d "${TMPDIR:-/tmp}/hf2q-release-binary-seal.XXXXXX")
trap 'rm -rf "$scratch"' EXIT

sha256_file() { shasum -a 256 "$1" | awk '{print $1}'; }
fail() {
  echo "$*" >&2
  exit 1
}

source_binary="$scratch/package/target/release/hf2q"
sealed_binary="$scratch/runner-temp/sealed/hf2q"
mkdir -p "$(dirname "$source_binary")"
printf '#!/usr/bin/env bash\nprintf "sealed candidate\\n"\n' >"$source_binary"
chmod 0755 "$source_binary"
expected_sha=$(sha256_file "$source_binary")

"$SEAL_SCRIPT" "$source_binary" "$sealed_binary" "$expected_sha" >/dev/null
[[ -x "$sealed_binary" ]] || fail "sealed copy is not executable"
[[ "$(sha256_file "$sealed_binary")" == "$expected_sha" ]] || \
  fail "sealed copy does not match the source digest"
[[ "$("$sealed_binary")" == "sealed candidate" ]] || \
  fail "sealed copy did not execute the expected candidate bytes"

# Reproduce the release failure: a later Cargo command relinks the original
# target path. The independently sealed executable must retain its identity.
printf '#!/usr/bin/env bash\nprintf "cargo relinked candidate\\n"\n' >"$source_binary"
chmod 0755 "$source_binary"
[[ "$(sha256_file "$source_binary")" != "$expected_sha" ]] || \
  fail "test setup did not mutate the Cargo target binary"
[[ "$(sha256_file "$sealed_binary")" == "$expected_sha" ]] || \
  fail "Cargo target mutation changed the sealed binary"
"$SEAL_SCRIPT" --verify "$sealed_binary" "$expected_sha" >/dev/null || \
  fail "identity verifier rejected the unchanged sealed binary"

chmod 0755 "$sealed_binary"
printf '#!/usr/bin/env bash\nprintf "tampered sealed candidate\\n"\n' >"$sealed_binary"
if "$SEAL_SCRIPT" --verify "$sealed_binary" "$expected_sha" >/dev/null 2>&1; then
  fail "identity verifier accepted a tampered sealed binary"
fi

wrong_destination="$scratch/runner-temp/wrong/hf2q"
if "$SEAL_SCRIPT" "$source_binary" "$wrong_destination" "$expected_sha" \
  >/dev/null 2>&1; then
  fail "seal accepted a source whose digest no longer matched"
fi
[[ ! -e "$wrong_destination" ]] || \
  fail "failed seal left a destination artifact"

# The standalone candidate freezes the packed-build bytes before signing, and
# the release revalidates the signed artifact instead of rebuilding it.
grep -qF 'cp "$built_binary" "$candidate_root/hf2q-unsigned"' \
  "$STANDALONE_WORKFLOW" || \
  fail "standalone workflow does not freeze the exact packed-build binary"
grep -qF 'unsigned_binary_sha256:$binary_sha256' "$STANDALONE_WORKFLOW" || \
  fail "standalone build receipt does not bind the unsigned binary digest"
grep -qF 'name: standalone-candidate-signed-${{ inputs.commit_sha }}' \
  "$STANDALONE_WORKFLOW" || \
  fail "standalone workflow does not publish the signed candidate separately"
grep -qF '[[ $(sha256_file "$binary") == "$binary_sha" ]]' "$VERIFY_SCRIPT" || \
  fail "release verifier does not recheck the signed binary digest"
grep -qF '.input.unsigned_sha256 == $unsigned_sha' "$VERIFY_SCRIPT" || \
  fail "release verifier does not bind signed bytes to the packed input"
grep -qF 'scripts/verify_standalone_candidate.sh \' "$RELEASE_WORKFLOW" || \
  fail "release workflow does not consume the frozen signed candidate"

echo "release binary seal contract: PASS"
