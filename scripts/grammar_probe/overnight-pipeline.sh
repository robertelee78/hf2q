#!/usr/bin/env bash
# All process identities, paths and commands belong in a reviewed local JSON.
set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
exec python3 -B "$SCRIPT_DIR/campaign_coordinator.py" "$@"
