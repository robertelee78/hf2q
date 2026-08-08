#!/usr/bin/env bash
set -euo pipefail

# Bind the shared OpenCode acceptance contract to Qwen 3.6. The running server
# must have accepted Qwen's embedded ChatML tool template at model load.
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

exec env \
  BASE_URL="${BASE_URL:-http://127.0.0.1:8081}" \
  MODEL="${MODEL:-qwen36-abliterix-t63-APEX}" \
  SENTINEL="${SENTINEL:-HF2Q_QWEN36_AGENTIC_OK}" \
  "$SCRIPT_DIR/test_deepseek4_agentic.sh"
