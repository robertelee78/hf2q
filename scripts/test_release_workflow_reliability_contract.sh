#!/usr/bin/env bash
# Release reliability contract: credential preflight and Trusted Publishing,
# pinned build image, draft-only asset replacement, no publish after a failed
# candidate, release-only checks in PR CI, and the operator retry script.
# Literal source probes intentionally contain unexpanded shell expressions.
# shellcheck disable=SC1003,SC2016
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)
RELEASE_WORKFLOW="$ROOT_DIR/.github/workflows/release.yml"
STANDALONE_WORKFLOW="$ROOT_DIR/.github/workflows/standalone-candidate.yml"
CI_WORKFLOW="$ROOT_DIR/.github/workflows/ci.yml"
VERIFY_SCRIPT="$ROOT_DIR/scripts/verify_standalone_candidate.sh"
APPLE_PREFLIGHT="$ROOT_DIR/scripts/preflight_apple_release_credentials.sh"
RELEASE_SCRIPT="$ROOT_DIR/scripts/release.sh"
scratch=$(mktemp -d "${TMPDIR:-/tmp}/hf2q-release-reliability.XXXXXX")
trap 'rm -rf "$scratch"' EXIT

fail() {
  echo "release workflow reliability contract: $*" >&2
  exit 1
}

workflow_job() {
  awk -v start="  ${2}:" '
    $0 == start { capture = 1 }
    capture && /^  [A-Za-z0-9_-]+:$/ && $0 != start { exit }
    capture { print }
  ' "$1"
}

bash -n "$APPLE_PREFLIGHT" "$RELEASE_SCRIPT" "$VERIFY_SCRIPT"
publish_job=$(workflow_job "$RELEASE_WORKFLOW" publish)
build_job=$(workflow_job "$STANDALONE_WORKFLOW" build-release-candidate)
sign_job=$(workflow_job "$STANDALONE_WORKFLOW" sign-release-candidate)

# --- Fix: publish never runs after a failed or missing candidate. ---
publish_if=$(grep -E '^    if: ' <<<"$publish_job") || fail "publish has no if condition"
grep -Fq "needs.standalone-candidate.result == 'success'" <<<"$publish_if" ||
  fail "publish can run after a failed standalone candidate"
grep -Fq 'always()' <<<"$publish_if" ||
  fail "publish no longer survives a skipped qualification job"

# --- Fix: Trusted Publishing, scoped OIDC, legacy-token fallback only. ---
grep -Eq 'uses: rust-lang/crates-io-auth-action@[0-9a-f]{40} # v' <<<"$publish_job" ||
  fail "crates-io-auth-action is not pinned by full commit SHA"
grep -Fq '      id-token: write' <<<"$publish_job" ||
  fail "publish job lacks the id-token permission Trusted Publishing needs"
[[ $(grep -c 'id-token' "$RELEASE_WORKFLOW") -eq 1 ]] ||
  fail "id-token permission must be granted to the publish job only"
for workflow in "$STANDALONE_WORKFLOW" "$CI_WORKFLOW"; do
  if grep -Fq 'id-token' "$workflow"; then
    fail "OIDC token permission leaked outside the publish job: $workflow"
  fi
done
grep -Fq '    environment: crates-publish' <<<"$publish_job" ||
  fail "publish job is not bound to the crates-publish environment"
auth_line=$(grep -nF 'uses: rust-lang/crates-io-auth-action@' <<<"$publish_job" | cut -d: -f1)
publish_line=$(grep -nF 'cargo publish --locked --no-verify' <<<"$publish_job" | cut -d: -f1)
[[ -n "$auth_line" && -n "$publish_line" && "$auth_line" -lt "$publish_line" ]] ||
  fail "Trusted Publishing exchange does not precede cargo publish"
between=$(sed -n "${auth_line},${publish_line}p" <<<"$publish_job")
[[ $(grep -c -- '- name: ' <<<"$between") -eq 1 ]] ||
  fail "Trusted Publishing exchange is not immediately before the publish step"
grep -Fq 'steps.crates-io-auth.outputs.token' <<<"$publish_job" ||
  fail "publish does not use the Trusted Publishing token"
grep -Fq 'falling back to CARGO_REGISTRY_TOKEN' <<<"$publish_job" ||
  fail "legacy token fallback is not logged"
if grep -Fq -- '--token' <<<"$publish_job"; then
  fail "registry token must not be passed on the cargo command line"
fi
run_blocks=$(grep -c '^        run: |$' <<<"$publish_job")
unset_lines=$(grep -c '^          unset ACTIONS_ID_TOKEN_REQUEST_TOKEN ACTIONS_ID_TOKEN_REQUEST_URL$' <<<"$publish_job")
[[ "$run_blocks" -gt 0 && "$run_blocks" -eq "$unset_lines" ]] ||
  fail "every publish run step must drop the OIDC request variables ($unset_lines/$run_blocks)"
if grep -E '^        run: [^|]' <<<"$publish_job"; then
  fail "single-line publish run steps cannot drop the OIDC request variables"
fi

# --- Fix: credential preflight is the first thing a Release does. ---
first_step=$(awk '/^    steps:$/ { s = 1; next } s && /^      - name: / { print; exit }' <<<"$build_job")
[[ "$first_step" == '      - name: Release credential preflight (crates.io)' ]] ||
  fail "crates.io credential preflight is not the first candidate build step"
preflight_step=$(awk '
  /^      - name: Release credential preflight \(crates.io\)$/ { s = 1 }
  s && /^      - name: Checkout exact candidate$/ { exit }
  s { print }
' <<<"$build_job")
grep -Fq "if: \${{ github.workflow == 'Release' }}" <<<"$preflight_step" ||
  fail "crates.io preflight is not limited to Release runs"
grep -Fq 'https://crates.io/api/v1/me/tokens' <<<"$preflight_step" ||
  fail "crates.io preflight does not authenticate the token"
grep -Eq '^ +401\|403\)$' <<<"$preflight_step" ||
  fail "crates.io preflight does not fail on a rejected token"
if grep -Fq 'CARGO_REGISTRY_TOKEN' <<<"${build_job/"$preflight_step"/}"; then
  fail "crates.io token is visible outside the preflight step"
fi
apple_line=$(grep -nF 'run: bash scripts/preflight_apple_release_credentials.sh' <<<"$sign_job" | cut -d: -f1)
download_line=$(grep -nF -- '- name: Download exact build candidate' <<<"$sign_job" | cut -d: -f1)
[[ -n "$apple_line" && -n "$download_line" && "$apple_line" -lt "$download_line" ]] ||
  fail "Apple credential preflight does not run before the candidate download"
for forbidden in unsigned candidate-build 'tar -' '--version'; do
  if grep -Fq -- "$forbidden" "$APPLE_PREFLIGHT"; then
    fail "Apple preflight must not touch candidate bytes: $forbidden"
  fi
done
for required in 'find-identity -v -p codesigning' '-checkend 0' \
  'notarytool history' 'delete-keychain'; do
  grep -Fq -- "$required" "$APPLE_PREFLIGHT" ||
    fail "Apple preflight omits: $required"
done
if grep -Fq 'list-keychains' "$APPLE_PREFLIGHT"; then
  fail "Apple preflight must not modify the user keychain search list"
fi

# --- Fix: pinned macOS image, recorded and asserted build host. ---
for workflow in "$RELEASE_WORKFLOW" "$STANDALONE_WORKFLOW"; do
  if grep -Fq 'macos-latest' "$workflow"; then
    fail "release-path workflow still floats on macos-latest: $workflow"
  fi
done
[[ $(grep -c '^    runs-on: macos-26$' "$STANDALONE_WORKFLOW") -eq 2 ]] ||
  fail "candidate build and sign jobs are not pinned to macos-26"
grep -Fq '    runs-on: macos-26' <<<"$publish_job" ||
  fail "publish job is not pinned to macos-26"
grep -Fq 'expected_build_macos_major=26' "$VERIFY_SCRIPT" ||
  fail "candidate verifier does not assert the pinned build macOS major"
for field in macos_product_version macos_build_version xcode_version \
  sdk_version runner_image_os runner_image_version; do
  grep -Fq "$field:\$$field" "$STANDALONE_WORKFLOW" ||
    fail "build receipt does not record $field"
  grep -Fq ".build_host.$field" "$VERIFY_SCRIPT" ||
    fail "candidate verifier does not check build_host.$field"
done
for source in "$STANDALONE_WORKFLOW" "$VERIFY_SCRIPT"; do
  grep -Fq 'schema_version == 2' "$source" ||
    fail "build receipt consumer is not on schema 2: $source"
done
grep -Fq 'schema_version:2,' "$STANDALONE_WORKFLOW" ||
  fail "build receipt producer is not on schema 2"

# --- Fix: published assets are immutable; stale draft assets are replaced. ---
grep -Fq 'gh release delete-asset "$tag" "$name" --yes' "$RELEASE_WORKFLOW" ||
  fail "draft asset replacement is missing"
awk '
  /^          ensure_exact_asset\(\) \{$/ { s = 1 }
  s { sub(/^          /, ""); print }
  s && /^}$/ { exit }
' "$RELEASE_WORKFLOW" >"$scratch/ensure_exact_asset.sh"
grep -q '^}$' "$scratch/ensure_exact_asset.sh" ||
  fail "could not extract ensure_exact_asset"

run_ensure() {
  local draft=$1 remote_content=$2
  local release_dir="$scratch/remote" calls="$scratch/calls"
  rm -rf "$release_dir" "$scratch/tmp"
  mkdir -p "$release_dir" "$scratch/tmp"
  : >"$calls"
  printf 'candidate\n' >"$scratch/asset.bin"
  if [[ -n "$remote_content" ]]; then
    printf '%s\n' "$remote_content" >"$release_dir/asset.bin"
  fi
  RUNNER_TEMP="$scratch/tmp" is_draft=$draft tag=v0.0.0 \
    RELEASE_DIR="$release_dir" CALLS="$calls" \
    bash -c '
      set -euo pipefail
      gh() {
        printf "%s\n" "$*" >>"$CALLS"
        case "$1 $2" in
          "release download")
            [[ -f "$RELEASE_DIR/$5" ]] || return 1
            cp "$RELEASE_DIR/$5" "$7/$5"
            ;;
          "release delete-asset") rm "$RELEASE_DIR/$4" ;;
          "release upload")
            [[ ! -e "$RELEASE_DIR/$(basename "$4")" ]] || return 1
            cp "$4" "$RELEASE_DIR/"
            ;;
          *) return 1 ;;
        esac
      }
      source "$0"
      ensure_exact_asset "$1"
    ' "$scratch/ensure_exact_asset.sh" "$scratch/asset.bin"
}

run_ensure true stale >/dev/null 2>&1 ||
  fail "draft release with a stale asset was not repaired"
cmp -s "$scratch/asset.bin" "$scratch/remote/asset.bin" ||
  fail "draft asset replacement did not upload the candidate bytes"
grep -q '^release delete-asset ' "$scratch/calls" ||
  fail "draft asset replacement did not delete the stale asset first"
if run_ensure false stale >/dev/null 2>&1; then
  fail "published release asset was replaced"
fi
if grep -q '^release \(delete-asset\|upload\) ' "$scratch/calls"; then
  fail "published release refusal still mutated assets"
fi
if run_ensure false "" >/dev/null 2>&1; then
  fail "published release accepted a missing asset"
fi
run_ensure true "" >/dev/null 2>&1 || fail "draft release did not accept an upload"
run_ensure false candidate >/dev/null 2>&1 ||
  fail "published release rejected identical bytes"
if grep -q '^release \(delete-asset\|upload\) ' "$scratch/calls"; then
  fail "identical published asset was rewritten"
fi

# --- Fix: release-only checks run in PR CI, in existing jobs only. ---
contracts_job=$(workflow_job "$CI_WORKFLOW" contracts)
release_ci_job=$(workflow_job "$CI_WORKFLOW" release)
grep -Fq 'cargo package --locked --no-verify' <<<"$contracts_job" ||
  fail "PR CI does not package the crate"
grep -Fq 'cd "$package_root"' <<<"$contracts_job" ||
  fail "PR CI does not check inside the packed root"
grep -Fq 'cargo check --locked --all-targets --all-features' <<<"$contracts_job" ||
  fail "PR CI does not check every target"
grep -Fq 'bash scripts/assert_macho_string_pool_aligned.sh \' <<<"$release_ci_job" ||
  fail "PR CI release job does not check LINKEDIT alignment"
[[ $(awk '/^jobs:$/ { s = 1; next } s && /^  [A-Za-z0-9_-]+:$/' "$CI_WORKFLOW" | wc -l | tr -d ' ') -eq 6 ]] ||
  fail "CI job set changed; ADR-061 forbids new CI jobs"

# --- Operator retry script. ---
for required in \
  "capacity_marker='not acquired by Runner'" \
  'gh run rerun "$run_id" --repo "$repo" --failed' \
  'check-runs/$job_id/annotations' \
  '--resume' \
  'FAILED step:'; do
  grep -Fq -- "$required" "$RELEASE_SCRIPT" ||
    fail "release.sh omits: $required"
done
if "$RELEASE_SCRIPT" >/dev/null 2>&1; then
  fail "release.sh accepted missing arguments"
fi
if "$RELEASE_SCRIPT" 0.1 deadbeef >/dev/null 2>&1; then
  fail "release.sh accepted a non-canonical version and SHA"
fi

echo "release workflow reliability contract: PASS"
