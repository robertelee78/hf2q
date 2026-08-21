#!/usr/bin/env bash
# Produce one Developer-ID-signed, notarized, thin-arm64 hf2q executable and
# a bounded proof receipt. The ZIP exists only to submit the standalone Mach-O
# to Apple's notary service; Apple cannot staple tickets to standalone binaries.
set -euo pipefail

if [[ $# -ne 6 ]]; then
  echo "usage: $0 INPUT_BINARY OUTPUT_DIRECTORY VERSION SOURCE_SHA TEAM_ID IDENTIFIER" >&2
  exit 2
fi

input_binary=$1
output_directory=$2
version=$3
source_sha=$4
team_id=$5
identifier=$6
asset_name=hf2q-aarch64-apple-darwin

sha256_file() { shasum -a 256 "$1" | awk '{print $1}'; }
fail() {
  echo "standalone release signing: $*" >&2
  exit 1
}

[[ $(uname -s) == Darwin && $(uname -m) == arm64 ]] || \
  fail "signing requires an Apple-Silicon macOS runner"
[[ -f "$input_binary" && -x "$input_binary" && ! -L "$input_binary" ]] || \
  fail "input must be a regular executable"
[[ "$output_directory" == /* ]] || fail "output directory must be absolute"
[[ ! -e "$output_directory" && ! -L "$output_directory" ]] || \
  fail "output directory already exists"
[[ "$version" =~ ^(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)$ ]] || \
  fail "version must be canonical stable SemVer"
[[ "$source_sha" =~ ^[0-9a-f]{40}$ ]] || fail "source SHA must be exact Git SHA-1"
[[ "$team_id" =~ ^[A-Z0-9]{10}$ ]] || fail "Apple Team ID is not canonical"
[[ "$identifier" =~ ^[A-Za-z0-9.-]+$ ]] || fail "signing identifier is not canonical"
[[ $(/usr/bin/lipo -archs "$input_binary" 2>/dev/null) == arm64 ]] || \
  fail "input is not an exact thin arm64 Mach-O"

signing_identity=${APPLE_DEVELOPER_ID_APPLICATION:?APPLE_DEVELOPER_ID_APPLICATION is required}
p12_base64=${APPLE_DEVELOPER_ID_APPLICATION_P12_BASE64:?APPLE_DEVELOPER_ID_APPLICATION_P12_BASE64 is required}
p12_password=${APPLE_DEVELOPER_ID_APPLICATION_P12_PASSWORD:?APPLE_DEVELOPER_ID_APPLICATION_P12_PASSWORD is required}
notary_key_base64=${APPLE_NOTARY_KEY_P8_BASE64:?APPLE_NOTARY_KEY_P8_BASE64 is required}
notary_key_id=${APPLE_NOTARY_KEY_ID:?APPLE_NOTARY_KEY_ID is required}
notary_issuer_id=${APPLE_NOTARY_ISSUER_ID:?APPLE_NOTARY_ISSUER_ID is required}
unset APPLE_DEVELOPER_ID_APPLICATION APPLE_DEVELOPER_ID_APPLICATION_P12_BASE64 \
  APPLE_DEVELOPER_ID_APPLICATION_P12_PASSWORD APPLE_NOTARY_KEY_P8_BASE64 \
  APPLE_NOTARY_KEY_ID APPLE_NOTARY_ISSUER_ID

[[ "$signing_identity" == "Developer ID Application: "*" ($team_id)" ]] || \
  fail "Developer ID identity does not bind the expected Team ID"
[[ "$signing_identity" != *$'\n'* && ${#signing_identity} -le 255 ]] || \
  fail "Developer ID identity is not canonical"
[[ "$notary_key_id" =~ ^[A-Z0-9]{10}$ ]] || fail "notary key ID is not canonical"
[[ "$notary_issuer_id" =~ ^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$ ]] || \
  fail "notary issuer ID is not canonical"
[[ $(HF2Q_NO_COMPLETION_INSTALL=1 "$input_binary" --version) == "hf2q $version" ]] || \
  fail "input version does not match the release version"
minimum_macos=$(/usr/bin/vtool -show-build "$input_binary" 2>/dev/null | \
  awk '$1 == "minos" {print $2}')
[[ "$minimum_macos" == 14.0 ]] || fail "input minimum macOS version is not exactly 14.0"

runner_temp=${RUNNER_TEMP:-${TMPDIR:-/tmp}}
[[ "$runner_temp" == /* && -d "$runner_temp" && ! -L "$runner_temp" ]] || \
  fail "RUNNER_TEMP must be an existing absolute directory"
if [[ -n ${APPLE_RELEASE_SECRET_DIRECTORY:-} ]]; then
  secret_directory=$APPLE_RELEASE_SECRET_DIRECTORY
  case "$secret_directory" in
    "$runner_temp"/hf2q-apple-release-secrets-*) ;;
    *) fail "Apple secret directory is outside the dedicated runner-temp namespace" ;;
  esac
  [[ ! -e "$secret_directory" && ! -L "$secret_directory" ]] || \
    fail "Apple secret directory already exists"
  mkdir -m 0700 "$secret_directory"
else
  secret_directory=$(mktemp -d "$runner_temp/hf2q-apple-release-secrets.XXXXXX")
fi
keychain="$secret_directory/release.keychain-db"
p12="$secret_directory/developer-id.p12"
notary_key="$secret_directory/AuthKey_${notary_key_id}.p8"
candidate="$secret_directory/hf2q"
submission_archive="$secret_directory/hf2q-notary.zip"
submission_json="$output_directory/notary-submission.json"
notary_wait="$output_directory/notary-wait.json"
notary_log="$secret_directory/notary-log.json"
gatekeeper_log="$secret_directory/gatekeeper.txt"
codesign_log="$secret_directory/codesign.txt"
keychain_password="$(/usr/bin/uuidgen)-$(/usr/bin/uuidgen)"

cleanup() {
  /usr/bin/security delete-keychain "$keychain" >/dev/null 2>&1 || true
  rm -f -- "$p12" "$notary_key" "$candidate" "$submission_archive" \
    "$notary_log" "$gatekeeper_log" "$codesign_log"
  rmdir "$secret_directory" >/dev/null 2>&1 || true
}
trap cleanup EXIT
trap 'exit 130' HUP INT TERM

umask 077
printf '%s' "$p12_base64" | /usr/bin/base64 -D >"$p12" || \
  fail "Developer ID secret is not valid base64"
printf '%s' "$notary_key_base64" | /usr/bin/base64 -D >"$notary_key" || \
  fail "notary key secret is not valid base64"
[[ -s "$p12" && -s "$notary_key" ]] || fail "Apple credential material is empty"
unset p12_base64 notary_key_base64

/usr/bin/security create-keychain -p "$keychain_password" "$keychain"
/usr/bin/security set-keychain-settings -lut 21600 "$keychain"
/usr/bin/security unlock-keychain -p "$keychain_password" "$keychain"
/usr/bin/security import "$p12" -k "$keychain" -P "$p12_password" \
  -T /usr/bin/codesign >/dev/null
/usr/bin/security set-key-partition-list -S apple-tool:,apple: -s \
  -k "$keychain_password" "$keychain" >/dev/null
unset p12_password

identities=$(/usr/bin/security find-identity -v -p codesigning "$keychain")
[[ $(grep -Fc -- "\"$signing_identity\"" <<<"$identities") -eq 1 ]] || \
  fail "ephemeral keychain does not contain exactly the intended signing identity"
[[ $(grep -Ec '^[[:space:]]*1 valid identities found$' <<<"$identities") -eq 1 ]] || \
  fail "ephemeral keychain contains an ambiguous signing identity set"
signing_fingerprint=$(grep -F -- "\"$signing_identity\"" <<<"$identities" | awk '{print $2}')
[[ "$signing_fingerprint" =~ ^[0-9A-F]{40}$ ]] || \
  fail "Developer ID signing fingerprint is not canonical"

/bin/cp "$input_binary" "$candidate"
/bin/chmod 0755 "$candidate"
/usr/bin/codesign --force --sign "$signing_fingerprint" --keychain "$keychain" \
  --identifier "$identifier" --options runtime --timestamp "$candidate"
/usr/bin/codesign --verify --strict --all-architectures --verbose=2 "$candidate"
/usr/bin/codesign --display --verbose=4 "$candidate" 2>"$codesign_log"
[[ $(grep -Fxc -- "Identifier=$identifier" "$codesign_log") -eq 1 ]] || \
  fail "signed identifier does not match"
[[ $(grep -Fxc -- "TeamIdentifier=$team_id" "$codesign_log") -eq 1 ]] || \
  fail "signed Team ID does not match"
[[ $(grep -Fxc -- "Authority=$signing_identity" "$codesign_log") -eq 1 ]] || \
  fail "signed authority does not match"
grep -Eq '^flags=0x[0-9a-f]+\(runtime\)' "$codesign_log" || \
  fail "hardened runtime is absent from the signature"
grep -Eq '^Timestamp=.+$' "$codesign_log" || \
  fail "secure timestamp is absent from the signature"
cdhash=$(sed -n 's/^CDHash=//p' "$codesign_log")
[[ "$cdhash" =~ ^[0-9a-f]{40,64}$ ]] || fail "signed CDHash is not canonical"
[[ $(grep -c '^CDHash=' "$codesign_log") -eq 1 ]] || fail "signed CDHash is ambiguous"
[[ $(HF2Q_NO_COMPLETION_INSTALL=1 "$candidate" --version) == "hf2q $version" ]] || \
  fail "signed candidate version changed"

/usr/bin/ditto -c -k --keepParent "$candidate" "$submission_archive"
archive_sha=$(sha256_file "$submission_archive")
mkdir -m 0700 "$output_directory"
output_binary="$output_directory/$asset_name"
/bin/cp "$candidate" "$output_binary"
/bin/chmod 0555 "$output_binary"
binary_size=$(stat -f '%z' "$output_binary")
binary_sha=$(sha256_file "$output_binary")

if ! /usr/bin/xcrun notarytool submit "$submission_archive" \
  --key "$notary_key" --key-id "$notary_key_id" --issuer "$notary_issuer_id" \
  --output-format json >"$submission_json"; then
  fail "Apple notarization upload did not complete successfully"
fi
submission_id=$(jq -er .id "$submission_json") || \
  fail "Apple notarization upload did not return a submission ID"
[[ "$submission_id" =~ ^[0-9a-fA-F-]{36}$ ]] || fail "notary submission ID is not canonical"
if ! /usr/bin/xcrun notarytool wait "$submission_id" \
  --key "$notary_key" --key-id "$notary_key_id" --issuer "$notary_issuer_id" \
  --timeout 30m --output-format json >"$notary_wait"; then
  fail "Apple notarization wait did not complete; resume the recorded submission ID"
fi
jq -e --arg submission_id "$submission_id" \
  'select(.id == $submission_id and .status == "Accepted")' \
  "$notary_wait" >/dev/null || \
  fail "Apple notarization status was not Accepted"
/usr/bin/xcrun notarytool log "$submission_id" \
  --key "$notary_key" --key-id "$notary_key_id" --issuer "$notary_issuer_id" \
  "$notary_log"
jq -e --arg cdhash "$cdhash" '
  .status == "Accepted"
  and ((.issues // []) | length) == 0
  and any(.ticketContents[]?;
    .digestAlgorithm == "SHA-256" and .cdhash == $cdhash)
' "$notary_log" >/dev/null || fail "notary log does not bind the accepted standalone binary"

accepted=0
for _ in $(seq 1 12); do
  if /usr/sbin/spctl --assess --type execute --verbose=4 "$candidate" \
    >"$gatekeeper_log" 2>&1 && \
    grep -Fq 'source=Notarized Developer ID' "$gatekeeper_log"; then
    accepted=1
    break
  fi
  sleep 5
done
[[ $accepted -eq 1 ]] || fail "Gatekeeper did not accept the notarized standalone binary"

submission_sha=$(sha256_file "$submission_json")
notary_wait_sha=$(sha256_file "$notary_wait")
notary_log_sha=$(sha256_file "$notary_log")
codesign_log_sha=$(sha256_file "$codesign_log")
gatekeeper_log_sha=$(sha256_file "$gatekeeper_log")

/bin/mv "$notary_log" "$output_directory/notary-log.json"
/bin/mv "$codesign_log" "$output_directory/codesign.txt"
/bin/mv "$gatekeeper_log" "$output_directory/gatekeeper.txt"
jq -nS \
  --arg source_sha "$source_sha" \
  --arg version "$version" \
  --arg asset_name "$asset_name" \
  --arg sha256 "$binary_sha" \
  --arg team_id "$team_id" \
  --arg identifier "$identifier" \
  --arg cdhash "$cdhash" \
  --arg submission_id "$submission_id" \
  --arg submission_archive_sha256 "$archive_sha" \
  --arg submission_sha256 "$submission_sha" \
  --arg wait_sha256 "$notary_wait_sha" \
  --arg notary_log_sha256 "$notary_log_sha" \
  --arg codesign_log_sha256 "$codesign_log_sha" \
  --arg gatekeeper_log_sha256 "$gatekeeper_log_sha" \
  --argjson size "$binary_size" \
  '{
    kind:"hf2q.standalone-apple-release-proof",
    schema_version:1,
    package:"hf2q",
    target:"aarch64-apple-darwin",
    source_sha:$source_sha,
    version:$version,
    asset:{name:$asset_name,size:$size,sha256:$sha256},
    signing:{
      authority:"Developer ID Application",
      team_id:$team_id,
      identifier:$identifier,
      cdhash:$cdhash,
      hardened_runtime:true,
      secure_timestamp:true
    },
    notarization:{
      status:"Accepted",
      submission_id:$submission_id,
      submission_archive_sha256:$submission_archive_sha256,
      submission_sha256:$submission_sha256,
      wait_sha256:$wait_sha256,
      log_sha256:$notary_log_sha256,
      standalone_ticket_stapled:false
    },
    verification:{
      codesign:"accepted",
      codesign_log_sha256:$codesign_log_sha256,
      gatekeeper:"Notarized Developer ID",
      gatekeeper_log_sha256:$gatekeeper_log_sha256
    }
  }' >"$output_directory/proof.json"
/bin/chmod 0444 "$output_directory"/{proof.json,notary-submission.json,notary-wait.json,notary-log.json,codesign.txt,gatekeeper.txt}
printf '%s  %s\n' "$binary_sha" "$asset_name" \
  >"$output_directory/${asset_name}.sha256"
/bin/chmod 0444 "$output_directory/${asset_name}.sha256"

for required in \
  "$asset_name" \
  "${asset_name}.sha256" \
  proof.json \
  notary-submission.json \
  notary-wait.json \
  notary-log.json \
  codesign.txt \
  gatekeeper.txt; do
  [[ -f "$output_directory/$required" && ! -L "$output_directory/$required" ]] || \
    fail "signed release output is incomplete or unsafe: $required"
done
[[ $(find "$output_directory" -mindepth 1 -maxdepth 1 -print | wc -l | tr -d ' ') == 8 ]] || \
  fail "signed release output contains unexpected entries"

printf '%s\n' "$output_binary"
