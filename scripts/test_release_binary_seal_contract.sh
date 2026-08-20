#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
SEAL_SCRIPT="$ROOT_DIR/scripts/seal_release_binary.sh"
CACHE_WORKFLOW="$ROOT_DIR/.github/workflows/cache-lifecycle.yml"
RELEASE_GATE="$ROOT_DIR/scripts/run_agentic_cache_release_gate.sh"
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

# shellcheck disable=SC2016
grep -qF 'sealed_binary="$release_evidence/hf2q-aarch64-apple-darwin"' \
  "$CACHE_WORKFLOW" || fail "cache workflow does not place the sealed binary outside Cargo target"
# shellcheck disable=SC2016
grep -qF '"$package_root/scripts/seal_release_binary.sh"' "$CACHE_WORKFLOW" || \
  fail "cache workflow does not invoke the packaged binary sealer"
# shellcheck disable=SC2016
grep -qF '"$signed_binary" "$sealed_binary" "$binary_sha"' "$CACHE_WORKFLOW" || \
  fail "cache workflow does not seal the signed candidate"
grep -qF "printf 'HF2Q_BIN=%s\\n' \"\$sealed_binary\"" "$CACHE_WORKFLOW" || \
  fail "cache workflow does not export the sealed binary"
# shellcheck disable=SC2016
grep -qF "printf 'EXPECTED_BINARY_SHA256=%s\\n' \"\$binary_sha\"" \
  "$CACHE_WORKFLOW" || fail "cache workflow does not export the signed-candidate digest"
# shellcheck disable=SC2016
grep -qF 'EXPECTED_BINARY_SHA256="$EXPECTED_BINARY_SHA256"' "$CACHE_WORKFLOW" || \
  fail "cache workflow does not pass the signed-candidate digest to the wrapper"

awk '
  /^start_server\(\)/ { in_start=1; next }
  in_start && /assert_exact_binary/ { asserted=1 }
  in_start && /ensure_guard_health/ { guarded=1; exit }
  END { exit(asserted && guarded ? 0 : 1) }
' "$RELEASE_GATE" || fail "release gate does not assert binary identity before model load"
# shellcheck disable=SC2016
grep -qF 'seal_release_binary.sh" --verify "$HF2Q_BIN" "$binary_sha"' \
  "$RELEASE_GATE" || fail "release launch guard does not use the tested identity verifier"
# shellcheck disable=SC2016
grep -qF 'binary_sha=$EXPECTED_BINARY_SHA256' "$RELEASE_GATE" || \
  fail "release wrapper does not keep signed-candidate digest authority"
# shellcheck disable=SC2016
if grep -qF 'binary_sha=$(sha256_file "$HF2Q_BIN")' "$RELEASE_GATE"; then
  fail "release wrapper adopts the sealed path digest as new authority"
fi

echo "release binary seal contract: PASS"
