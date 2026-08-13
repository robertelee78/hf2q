#!/usr/bin/env bash
# Copy one built hf2q executable out of Cargo's mutable target directory and
# bind the copy to the digest used by the hardware release receipt.
set -euo pipefail

sha256_file() { shasum -a 256 "$1" | awk '{print $1}'; }

if [[ "${1:-}" == --verify ]]; then
  [[ $# -eq 3 ]] || {
    echo "usage: $0 --verify SEALED_BINARY EXPECTED_SHA256" >&2
    exit 2
  }
  sealed_binary=$2
  expected_sha256=$3
  [[ "$expected_sha256" =~ ^[0-9a-f]{64}$ ]] || {
    echo "expected binary SHA-256 must be a lowercase 64-character digest" >&2
    exit 2
  }
  [[ -x "$sealed_binary" ]] || {
    echo "release binary is missing or non-executable: $sealed_binary" >&2
    exit 1
  }
  actual_sha256=$(sha256_file "$sealed_binary")
  [[ "$actual_sha256" == "$expected_sha256" ]] || {
    echo "release binary digest changed: expected $expected_sha256, got $actual_sha256 ($sealed_binary)" >&2
    exit 1
  }
  printf '%s\n' "$sealed_binary"
  exit 0
fi

if [[ $# -ne 3 ]]; then
  echo "usage: $0 SOURCE_BINARY SEALED_BINARY EXPECTED_SHA256" >&2
  exit 2
fi

source_binary=$1
sealed_binary=$2
expected_sha256=$3
sealed_parent=$(dirname "$sealed_binary")
temporary_binary="${sealed_binary}.tmp.$$"

[[ "$expected_sha256" =~ ^[0-9a-f]{64}$ ]] || {
  echo "expected binary SHA-256 must be a lowercase 64-character digest" >&2
  exit 2
}
[[ -x "$source_binary" ]] || {
  echo "source release binary is not executable: $source_binary" >&2
  exit 2
}
[[ "$source_binary" != "$sealed_binary" ]] || {
  echo "sealed release binary must use a distinct path" >&2
  exit 2
}
[[ ! -e "$sealed_binary" && ! -L "$sealed_binary" ]] || {
  echo "sealed release binary path already exists: $sealed_binary" >&2
  exit 2
}

source_sha_before=$(sha256_file "$source_binary")
[[ "$source_sha_before" == "$expected_sha256" ]] || {
  echo "source release binary digest mismatch: expected $expected_sha256, got $source_sha_before" >&2
  exit 1
}

mkdir -p "$sealed_parent"
trap 'rm -f "$temporary_binary"' EXIT
cp "$source_binary" "$temporary_binary"
chmod 0555 "$temporary_binary"

source_sha_after=$(sha256_file "$source_binary")
temporary_sha=$(sha256_file "$temporary_binary")
[[ "$source_sha_after" == "$expected_sha256" ]] || {
  echo "source release binary changed while it was being sealed" >&2
  exit 1
}
[[ "$temporary_sha" == "$expected_sha256" ]] || {
  echo "sealed release binary copy digest mismatch" >&2
  exit 1
}

mv "$temporary_binary" "$sealed_binary"
trap - EXIT
[[ -x "$sealed_binary" ]] || {
  echo "sealed release binary is not executable: $sealed_binary" >&2
  exit 1
}
sealed_sha=$(sha256_file "$sealed_binary")
[[ "$sealed_sha" == "$expected_sha256" ]] || {
  echo "sealed release binary final digest mismatch" >&2
  exit 1
}

printf '%s\n' "$sealed_binary"
