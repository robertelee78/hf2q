#!/usr/bin/env bash
set -euo pipefail

# Bind the shared OpenCode acceptance contract to Gemma 4. The running server
# must have accepted Gemma's embedded native tool template at model load.
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

exec env \
  BASE_URL="${BASE_URL:-http://127.0.0.1:8082}" \
  MODEL="${MODEL:-Gemma4 Ara 2pass Baseline}" \
  SENTINEL="${SENTINEL:-HF2Q_GEMMA4_AGENTIC_OK}" \
  "$SCRIPT_DIR/test_deepseek4_agentic.sh"
