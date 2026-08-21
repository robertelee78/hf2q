#!/usr/bin/env bash
set -euo pipefail

# Bind the shared OpenCode acceptance contract to Gemma 4. The running server
# must have accepted Gemma's embedded native tool template at model load.
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

# Four full-context Gemma slots process a matched ~24K-token resumed batch in
# the same 13-14 second envelope as the peer on the target M5 Max. Keep
# single/cached-turn TTFT strict while giving that peer-bound batch a small
# thermal/jitter margin.
exec env \
  BASE_URL="${BASE_URL:-http://127.0.0.1:8082}" \
  MODEL="${MODEL:-Gemma4 Ara 2pass Baseline}" \
  MAX_TOKENS="${MAX_TOKENS:-512}" \
  MAX_CACHED_RESPONSE_MS="${MAX_CACHED_RESPONSE_MS:-16000}" \
  SENTINEL="${SENTINEL:-HF2Q_GEMMA4_AGENTIC_OK}" \
  "$SCRIPT_DIR/test_deepseek4_agentic.sh"
