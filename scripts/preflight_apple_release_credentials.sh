#!/usr/bin/env bash
# Fail-fast Apple release credential preflight for the protected
# apple-release sign job. It proves, without printing any secret, that:
#   * every signing/notary secret and variable is present and canonical,
#   * the Developer ID .p12 imports with its password into an ephemeral
#     keychain and yields exactly one valid (unexpired) codesigning identity
#     matching APPLE_DEVELOPER_ID_APPLICATION,
#   * the certificate does not expire within the warning window,
#   * the App Store Connect notary API key authenticates (notarytool history).
# It never receives or executes candidate bytes (ADR-045 signer boundary) and
# never touches the user keychain search list.
set -euo pipefail

fail() {
  echo "::error::Apple release credential preflight: $*" >&2
  exit 1
}

[[ $(uname -s) == Darwin ]] || fail "requires macOS"
missing=()
for name in APPLE_DEVELOPER_ID_APPLICATION \
  APPLE_DEVELOPER_ID_APPLICATION_P12_BASE64 \
  APPLE_DEVELOPER_ID_APPLICATION_P12_PASSWORD \
  APPLE_NOTARY_KEY_P8_BASE64 APPLE_NOTARY_KEY_ID APPLE_NOTARY_ISSUER_ID \
  APPLE_TEAM_ID APPLE_CODESIGN_IDENTIFIER; do
  [[ -n ${!name:-} ]] || missing+=("$name")
done
[[ ${#missing[@]} -eq 0 ]] || \
  fail "missing apple-release secrets/variables: ${missing[*]}"

signing_identity=$APPLE_DEVELOPER_ID_APPLICATION
p12_base64=$APPLE_DEVELOPER_ID_APPLICATION_P12_BASE64
p12_password=$APPLE_DEVELOPER_ID_APPLICATION_P12_PASSWORD
notary_key_base64=$APPLE_NOTARY_KEY_P8_BASE64
notary_key_id=$APPLE_NOTARY_KEY_ID
notary_issuer_id=$APPLE_NOTARY_ISSUER_ID
team_id=$APPLE_TEAM_ID
unset APPLE_DEVELOPER_ID_APPLICATION_P12_BASE64 \
  APPLE_DEVELOPER_ID_APPLICATION_P12_PASSWORD APPLE_NOTARY_KEY_P8_BASE64

[[ "$team_id" =~ ^[A-Z0-9]{10}$ ]] || fail "APPLE_TEAM_ID is not canonical"
[[ "$APPLE_CODESIGN_IDENTIFIER" == us.hf2q.cli ]] || \
  fail "APPLE_CODESIGN_IDENTIFIER is not us.hf2q.cli"
[[ "$signing_identity" == "Developer ID Application: "*" ($team_id)" ]] || \
  fail "APPLE_DEVELOPER_ID_APPLICATION does not bind APPLE_TEAM_ID"
[[ "$notary_key_id" =~ ^[A-Z0-9]{10}$ ]] || fail "APPLE_NOTARY_KEY_ID is not canonical"
[[ "$notary_issuer_id" =~ ^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$ ]] || \
  fail "APPLE_NOTARY_ISSUER_ID is not canonical"

# Warn (not fail) inside this window so the owner can renew before expiry.
warn_seconds=$((60 * 24 * 60 * 60))
runner_temp=${RUNNER_TEMP:-${TMPDIR:-/tmp}}
scratch=$(mktemp -d "$runner_temp/hf2q-apple-preflight.XXXXXX")
keychain="$scratch/preflight.keychain-db"
cleanup() {
  /usr/bin/security delete-keychain "$keychain" >/dev/null 2>&1 || true
  rm -rf -- "$scratch"
}
trap cleanup EXIT
trap 'exit 130' HUP INT TERM

umask 077
p12="$scratch/developer-id.p12"
notary_key="$scratch/AuthKey_${notary_key_id}.p8"
printf '%s' "$p12_base64" | /usr/bin/base64 -D >"$p12" 2>/dev/null || \
  fail "APPLE_DEVELOPER_ID_APPLICATION_P12_BASE64 is not valid base64"
printf '%s' "$notary_key_base64" | /usr/bin/base64 -D >"$notary_key" 2>/dev/null || \
  fail "APPLE_NOTARY_KEY_P8_BASE64 is not valid base64"
unset p12_base64 notary_key_base64
[[ -s "$p12" ]] || fail "Developer ID .p12 is empty"
grep -Fq -- '-----BEGIN PRIVATE KEY-----' "$notary_key" || \
  fail "notary key is not a PEM private key"

keychain_password="$(/usr/bin/uuidgen)-$(/usr/bin/uuidgen)"
/usr/bin/security create-keychain -p "$keychain_password" "$keychain"
/usr/bin/security unlock-keychain -p "$keychain_password" "$keychain"
/usr/bin/security import "$p12" -k "$keychain" -P "$p12_password" \
  >/dev/null 2>&1 || \
  fail "Developer ID .p12 does not import with APPLE_DEVELOPER_ID_APPLICATION_P12_PASSWORD"
unset p12_password

certificate="$scratch/developer-id.pem"
/usr/bin/security find-certificate -c "$signing_identity" -p "$keychain" \
  >"$certificate" 2>/dev/null || \
  fail "the .p12 does not contain the APPLE_DEVELOPER_ID_APPLICATION certificate"
not_after=$(/usr/bin/openssl x509 -in "$certificate" -noout -enddate | \
  sed 's/^notAfter=//')
/usr/bin/openssl x509 -in "$certificate" -noout -checkend 0 >/dev/null || \
  fail "Developer ID certificate expired ($not_after)"
if ! /usr/bin/openssl x509 -in "$certificate" -noout \
  -checkend "$warn_seconds" >/dev/null; then
  echo "::warning::Developer ID certificate expires within 60 days ($not_after); renew it"
fi
identities=$(/usr/bin/security find-identity -v -p codesigning "$keychain")
[[ $(grep -Fc -- "\"$signing_identity\"" <<<"$identities") -eq 1 ]] || \
  fail "the .p12 has no valid codesigning identity for APPLE_DEVELOPER_ID_APPLICATION (revoked, expired, or wrong certificate)"

/usr/bin/xcrun notarytool history \
  --key "$notary_key" --key-id "$notary_key_id" --issuer "$notary_issuer_id" \
  --output-format json >/dev/null 2>"$scratch/notary.err" || {
  sed 's/^/notarytool: /' "$scratch/notary.err" >&2
  fail "App Store Connect notary API key does not authenticate"
}

echo "Apple release credential preflight: PASS (Developer ID valid until $not_after; notary key authenticates)"
