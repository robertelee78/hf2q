#!/usr/bin/env bash
set -euo pipefail
# Requires an externally guarded, dedicated fresh Qwen38 candidate server.
: "${SERVER_LOG:?actual dedicated server stderr required}"
: "${SHARED_LIFECYCLE_EXCLUSIVE:?set1 only after exclusive-case review}"
 : "${OUT_DIR:?fresh absolute artifact directory required}"
[[ "$OUT_DIR" == /* && ! -e "$OUT_DIR" ]] || {
  echo "OUT_DIR must be absolute and not already exist" >&2; exit 2;
}
umask 077
mkdir -m 700 "$OUT_DIR"
export OUT_DIR
export SHARED_CANCELLATION=1
export CONTEXT_LINES=${CONTEXT_LINES:-128}
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
exec bash "$SCRIPT_DIR/test_agentic_cache_lifecycle.sh"
