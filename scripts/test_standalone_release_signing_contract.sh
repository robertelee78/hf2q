#!/usr/bin/env bash
# Literal source-contract probes intentionally contain shell expressions that
# must not expand in this test process.
# shellcheck disable=SC2016
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)
SIGN_SCRIPT="$ROOT_DIR/scripts/sign_notarize_standalone_release.sh"
CACHE_WORKFLOW="$ROOT_DIR/.github/workflows/cache-lifecycle.yml"
RELEASE_WORKFLOW="$ROOT_DIR/.github/workflows/release.yml"

fail() {
  echo "$*" >&2
  exit 1
}

bash -n "$SIGN_SCRIPT"
if "$SIGN_SCRIPT" >/dev/null 2>&1; then
  fail "signing script accepted a missing release contract"
fi

for required in \
  '/usr/bin/codesign --force --sign' \
  '--identifier "$identifier" --options runtime --timestamp' \
  '/usr/bin/ditto -c -k --keepParent' \
  'notarytool submit "$submission_archive"' \
  'notarytool wait "$submission_id"' \
  '--timeout 30m --output-format json' \
  'notarytool log "$submission_id"' \
  'source=Notarized Developer ID' \
  'standalone_ticket_stapled:false'; do
  grep -Fq -- "$required" "$SIGN_SCRIPT" || \
    fail "signing script is missing required contract: $required"
done

if rg -n 'stapler[[:space:]]+staple' "$SIGN_SCRIPT"; then
  fail "standalone Mach-O must not claim an unsupported stapled ticket"
fi
grep -Fq 'environment: apple-release' "$CACHE_WORKFLOW" || \
  fail "exact-artifact gate is not protected by the Apple release environment"
grep -Fq 'scripts/sign_notarize_standalone_release.sh' "$CACHE_WORKFLOW" || \
  fail "exact-artifact gate does not sign before hardware validation"
grep -Fq 'sealed_binary="$release_evidence/hf2q-aarch64-apple-darwin"' \
  "$CACHE_WORKFLOW" || \
  fail "exact-artifact gate does not export the tested signed binary"
grep -Fq 'standalone_proof="$standalone_dir/proof.json"' "$RELEASE_WORKFLOW" || \
  fail "release workflow does not consume the signed proof receipt"
grep -Fq 'hf2q-aarch64-apple-darwin' "$RELEASE_WORKFLOW" || \
  fail "release workflow does not publish the native standalone asset"
if rg -n 'gh release upload.*--clobber|--clobber.*hf2q-aarch64-apple-darwin' \
  "$RELEASE_WORKFLOW"; then
  fail "immutable standalone assets must never be clobbered"
fi

echo "standalone release signing contract: PASS"
