#!/usr/bin/env bash
# Literal source-contract probes intentionally contain shell expressions that
# must not expand in this test process.
# shellcheck disable=SC1003,SC2016
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)
SIGN_SCRIPT="$ROOT_DIR/scripts/sign_notarize_standalone_release.sh"
CACHE_WORKFLOW="$ROOT_DIR/.github/workflows/cache-lifecycle.yml"
RELEASE_WORKFLOW="$ROOT_DIR/.github/workflows/release.yml"

fail() {
  echo "$*" >&2
  exit 1
}

workflow_executes_unsigned_candidate() {
  local source=$1
  grep -Eq \
    '^[[:space:]]*(env[[:space:]]+)?([A-Za-z_][A-Za-z0-9_]*=[^[:space:]]+[[:space:]]+)*"\$unsigned"([[:space:]]|$)' \
    <<<"$source" || \
    grep -Eq \
      '\$\([[:space:]]*(env[[:space:]]+)?([A-Za-z_][A-Za-z0-9_]*=[^[:space:]]+[[:space:]]+)*"\$unsigned"([[:space:]]|$)' \
      <<<"$source"
}

bash -n "$SIGN_SCRIPT"
if "$SIGN_SCRIPT" >/dev/null 2>&1; then
  fail "signing script accepted a missing release contract"
fi

for required in \
  '/usr/bin/codesign --force --sign' \
  '/usr/bin/security list-keychains -d user -s "$keychain"' \
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
grep -Fq '"${original_user_keychains[@]}" >/dev/null 2>&1 || true' \
  "$SIGN_SCRIPT" || \
  fail "signing cleanup does not restore the original user keychain search list"

runtime_flag_pattern='^CodeDirectory .* flags=0x[0-9a-f]+\(runtime\)( |$)'
grep -Fq "grep -Eq '$runtime_flag_pattern'" "$SIGN_SCRIPT" || \
  fail "signing script does not inspect runtime flags in the CodeDirectory line"
valid_codesign_metadata='CodeDirectory v=20500 size=123 flags=0x10000(runtime) hashes=42+7 location=embedded'
grep -Eq "$runtime_flag_pattern" <<<"$valid_codesign_metadata" || \
  fail "runtime verifier rejected Apple's CodeDirectory output shape"
invalid_codesign_metadata='CodeDirectory v=20500 size=123 flags=0x0(none) hashes=42+7 location=embedded'
if grep -Eq "$runtime_flag_pattern" <<<"$invalid_codesign_metadata"; then
  fail "runtime verifier accepted a signature without hardened runtime"
fi

if grep -En 'stapler[[:space:]]+staple' "$SIGN_SCRIPT"; then
  fail "standalone Mach-O must not claim an unsupported stapled ticket"
fi
grep -Fq 'environment: apple-release' "$CACHE_WORKFLOW" || \
  fail "exact-artifact gate is not protected by the Apple release environment"
grep -Fq 'signer="$GITHUB_WORKSPACE/scripts/sign_notarize_standalone_release.sh"' \
  "$CACHE_WORKFLOW" || \
  fail "protected signing does not invoke the exact checked-out signer"
grep -Fq '"$signer" \' "$CACHE_WORKFLOW" || \
  fail "protected signing does not call its verified checkout signer"
sign_job=$(sed -n '/^  sign-release-candidate:/,/^  exact-artifact-cache-lifecycle:/p' \
  "$CACHE_WORKFLOW")
if grep -Fq 'tar -xzf' <<<"$sign_job" || \
  grep -Fq '$package_root/scripts/sign_notarize_standalone_release.sh' <<<"$sign_job"; then
  fail "protected signing must not execute code extracted from the unsigned artifact"
fi
sign_job_without_signer_call=$(sed \
  '/^[[:space:]]*"\$signer" \\$/,/^[[:space:]]*"\$team_id" "\$APPLE_CODESIGN_IDENTIFIER"$/d' \
  <<<"$sign_job")
if workflow_executes_unsigned_candidate "$sign_job_without_signer_call"; then
  fail "protected signing job must not execute the unsigned candidate"
fi
workflow_execution_mutant='          "$unsigned" --version'
if ! workflow_executes_unsigned_candidate "$workflow_execution_mutant"; then
  fail "unsigned-candidate execution detector accepted a direct execution mutant"
fi
if grep -Fq -- '--version' "$SIGN_SCRIPT"; then
  fail "secret-bearing signer must not execute candidate version commands"
fi
if grep -En \
  '^[[:space:]]*(env[[:space:]]+)?([A-Za-z_][A-Za-z0-9_]*=[^[:space:]]+[[:space:]]+)*"\$(input_binary|candidate)"([[:space:]]|$)' \
  "$SIGN_SCRIPT"; then
  fail "secret-bearing signer must not execute candidate bytes"
fi
if grep -En \
  '\$\([[:space:]]*(env[[:space:]]+)?([A-Za-z_][A-Za-z0-9_]*=[^[:space:]]+[[:space:]]+)*"\$(input_binary|candidate)"([[:space:]]|$)' \
  "$SIGN_SCRIPT"; then
  fail "secret-bearing signer must not execute candidate bytes in a substitution"
fi
grep -Fq 'sealed_binary="$release_evidence/hf2q-aarch64-apple-darwin"' \
  "$CACHE_WORKFLOW" || \
  fail "exact-artifact gate does not export the tested signed binary"
grep -Fq 'standalone_proof="$standalone_dir/proof.json"' "$RELEASE_WORKFLOW" || \
  fail "release workflow does not consume the signed proof receipt"
grep -Fq 'hf2q-aarch64-apple-darwin' "$RELEASE_WORKFLOW" || \
  fail "release workflow does not publish the native standalone asset"
if grep -En 'gh release upload.*--clobber|--clobber.*hf2q-aarch64-apple-darwin' \
  "$RELEASE_WORKFLOW"; then
  fail "immutable standalone assets must never be clobbered"
fi
if [[ $(grep -Fc '"$proof_install/hf2q" --state-root "$proof_state" \' \
  "$RELEASE_WORKFLOW") -ne 4 ]]; then
  fail "local and public installer proofs must run and revalidate installed setup"
fi
if grep -Fq "printf 'operator config" "$RELEASE_WORKFLOW"; then
  fail "release proof must not pre-seed a placeholder operator config"
fi
if [[ $(grep -Fc 'proof_home=$(cd "$(mktemp -d "$RUNNER_TEMP/hf2q-' \
  "$RELEASE_WORKFLOW") -ne 2 ]]; then
  fail "local and public installer proofs must use physical isolated home paths"
fi
for required in \
  'test -s "$proof_state/config.toml"' \
  'grep -Fx '\''schema_version = 2'\'' "$proof_state/config.toml"' \
  'cmp -s "$proof_state/config.toml" "$setup_golden"' \
  'model_sha=$(shasum -a 256 "$proof_state/models/model.gguf" | awk '\''{print $1}'\'')' \
  'test ! -e "$proof_install/.hf2q-standalone.json"' \
  'test ! -e "$proof_install/.hf2q-standalone.lock"' \
  'test ! -e "$proof_install/.hf2q-previous"' \
  'test -f "$proof_state/config.toml"' \
  'test -f "$proof_state/models/model.gguf"'; do
  if [[ $(grep -Fc -- "$required" "$RELEASE_WORKFLOW") -ne 2 ]]; then
    fail "local and public installer proofs are missing: $required"
  fi
done

echo "standalone release signing contract: PASS"
