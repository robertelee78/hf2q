#!/usr/bin/env bash
# adr005_mteb_sanity.sh — ADR-005 Phase 2b MTEB 5-task sanity gate operator recipe.
#
# Drives the env-gated test in tests/mteb_sanity_harness.rs:
#   * mteb_sanity_matrix_5_tasks_3_models — for each of 3 day-one
#     BERT-family GGUFs, spawn `hf2q serve --embedding-model <gguf>`,
#     run a Python `mteb` runner over BIOSSES, Banking77Classification,
#     NFCorpus, TwentyNewsgroupsClustering, SciDocsRR, and assert
#     |measured - expected| <= 1.0 per cell (AC line 3979).
#
# Closure: 15/15 cells PASS = AC line 3979 closes. Any FAIL = a
# concrete embedding-quality regression (or an expected_scores.json
# drift; re-pin via PR with verified leaderboard values).
#
# Required env (existence-checked at script start; no sentinel fallback
# per pattern_env_override_close_gates):
#   HF2Q_MTEB_GGUF_NOMIC   — nomic-embed-text-v1.5 GGUF
#   HF2Q_MTEB_GGUF_MXBAI   — mxbai-embed-large-v1 GGUF
#   HF2Q_MTEB_GGUF_BGE     — bge-small-en-v1.5 GGUF
#
# Optional env:
#   HF2Q_MTEB_PYTHON       — python3 binary (default: python3 from PATH)
#   HF2Q_MTEB_VENV         — venv path (default: /tmp/hf2q-mteb-venv)
#   HF2Q_MTEB_PORT_BASE    — hf2q listen port (default: 8765)
#   HF2Q_MTEB_FLOOR        — per-cell drift floor in pts (default: 1.0)
#   HF2Q_MTEB_ALLOW_WARN   — accept WARN(no-baseline) cells (default: 0)
#                            Set =1 only while expected_scores.json
#                            still has null placeholders for nomic/mxbai.
#
# Usage:
#   scripts/adr005_mteb_sanity.sh                # default: full live matrix
#   scripts/adr005_mteb_sanity.sh --rebuild      # force `cargo build --release`
#   scripts/adr005_mteb_sanity.sh --venv /path   # pin a custom venv path
#   scripts/adr005_mteb_sanity.sh --skip-venv    # use system python (deps must already be installed)
#
# Exit codes:
#   0  all gates PASS (or all PASS + WARN under HF2Q_MTEB_ALLOW_WARN=1)
#   1  usage / arg error
#   2  cargo test exited non-zero (gate FAIL or test crash)
#   3  prerequisite missing (binary not built, GGUF not found, etc.)
#   4  venv setup failed

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"

VENV_PATH="${HF2Q_MTEB_VENV:-/tmp/hf2q-mteb-venv}"
PYTHON_BIN="${HF2Q_MTEB_PYTHON:-python3}"
PORT_BASE="${HF2Q_MTEB_PORT_BASE:-8765}"
FLOOR="${HF2Q_MTEB_FLOOR:-1.0}"
ALLOW_WARN="${HF2Q_MTEB_ALLOW_WARN:-0}"
REBUILD=0
SKIP_VENV=0

usage() {
  cat <<EOF
Usage: scripts/adr005_mteb_sanity.sh [--rebuild] [--venv PATH] [--skip-venv]

Drives ADR-005 Phase 2b MTEB sanity (15-cell matrix; AC line 3979).

  --rebuild         force cargo build --release before running tests
  --venv PATH       venv path (default: \$HF2Q_MTEB_VENV or /tmp/hf2q-mteb-venv)
  --skip-venv       use \$HF2Q_MTEB_PYTHON directly (deps already installed)
  -h, --help        show this help

Exit codes: 0=PASS  1=usage  2=test-fail  3=prereq-missing  4=venv-fail
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help) usage; exit 0 ;;
    --rebuild) REBUILD=1; shift ;;
    --venv)
      [[ $# -ge 2 ]] || { echo "error: --venv requires a path" >&2; exit 1; }
      VENV_PATH="$2"; shift 2 ;;
    --skip-venv) SKIP_VENV=1; shift ;;
    -*) usage >&2; echo "error: unknown flag: $1" >&2; exit 1 ;;
    *)  usage >&2; echo "error: unexpected positional arg: $1" >&2; exit 1 ;;
  esac
done

echo "=== ADR-005 Phase 2b — MTEB 5-task sanity gate ==="
echo "REPO_ROOT:       $REPO_ROOT"
echo "venv:            $([ "$SKIP_VENV" -eq 1 ] && echo "(skipped) $PYTHON_BIN" || echo "$VENV_PATH")"
echo "port base:       $PORT_BASE"
echo "per-cell floor:  $FLOOR pts"
echo "allow-warn:      $ALLOW_WARN"
echo

# 1. Existence-check on every GGUF env var. Fails loud — no sentinel
#    fallback per pattern_env_override_close_gates.
check_gguf_env() {
  local var_name="$1"
  local val="${!var_name:-}"
  if [[ -z "$val" ]]; then
    echo "error: $var_name is not set; required for the matrix" >&2
    return 3
  fi
  if [[ ! -f "$val" ]]; then
    echo "error: $var_name=$val does not exist (or is not a regular file)" >&2
    return 3
  fi
  echo "  $var_name = $val"
  return 0
}

echo "--- GGUF existence check ---"
for var in HF2Q_MTEB_GGUF_NOMIC HF2Q_MTEB_GGUF_MXBAI HF2Q_MTEB_GGUF_BGE; do
  if ! check_gguf_env "$var"; then exit 3; fi
done
echo

# 2. Build the hf2q binary if missing or --rebuild.
HF2Q_BIN="$REPO_ROOT/target/release/hf2q"
if [[ "$REBUILD" -eq 1 ]] || [[ ! -x "$HF2Q_BIN" ]]; then
  echo "--- cargo build --release --bin hf2q ---"
  ( cd "$REPO_ROOT" && cargo build --release --bin hf2q )
fi
if [[ ! -x "$HF2Q_BIN" ]]; then
  echo "error: hf2q binary not found at $HF2Q_BIN after build" >&2
  exit 3
fi
echo "  hf2q bin: $HF2Q_BIN"
echo

# 3. Set up the Python venv with mteb + numpy + requests pinned via
#    tests/fixtures/mteb/requirements.txt.
if [[ "$SKIP_VENV" -eq 1 ]]; then
  echo "--- venv SKIPPED (--skip-venv); using $PYTHON_BIN directly ---"
  RUNNER_PYTHON="$PYTHON_BIN"
else
  echo "--- venv setup at $VENV_PATH ---"
  if [[ ! -d "$VENV_PATH" ]]; then
    "$PYTHON_BIN" -m venv "$VENV_PATH" || { echo "error: venv create failed" >&2; exit 4; }
  fi
  # shellcheck disable=SC1090,SC1091
  source "$VENV_PATH/bin/activate"
  python -m pip install --quiet --upgrade pip || { echo "error: pip upgrade failed" >&2; exit 4; }
  python -m pip install --quiet -r "$REPO_ROOT/tests/fixtures/mteb/requirements.txt" \
    || { echo "error: pip install requirements.txt failed" >&2; exit 4; }
  RUNNER_PYTHON="$VENV_PATH/bin/python"
  echo "  runner python: $RUNNER_PYTHON"
  python -c "import mteb, numpy, requests; print(f'  mteb={mteb.__version__} numpy={numpy.__version__} requests={requests.__version__}')"
fi
echo

# 4. Run the gated cargo test. The test reads the same env knobs the
#    script just validated, so we forward them all explicitly.
echo "--- cargo test --release --test mteb_sanity_harness ---"
echo "    HF2Q_MTEB_E2E=1"
echo "    HF2Q_MTEB_PYTHON=$RUNNER_PYTHON"
echo "    HF2Q_MTEB_PORT_BASE=$PORT_BASE"
echo "    HF2Q_MTEB_FLOOR=$FLOOR"
echo "    HF2Q_MTEB_ALLOW_WARN=$ALLOW_WARN"
echo

cd "$REPO_ROOT"

EXIT_CODE=0
if HF2Q_MTEB_E2E=1 \
    HF2Q_MTEB_PYTHON="$RUNNER_PYTHON" \
    HF2Q_MTEB_PORT_BASE="$PORT_BASE" \
    HF2Q_MTEB_FLOOR="$FLOOR" \
    HF2Q_MTEB_ALLOW_WARN="$ALLOW_WARN" \
    cargo test --release --test mteb_sanity_harness \
      -- --test-threads=1 --nocapture; then
  echo
  echo "=== ADR-005 Phase 2b — MTEB sanity gate PASS ==="
  EXIT_CODE=0
else
  rc=$?
  echo
  echo "=== ADR-005 Phase 2b — MTEB sanity gate FAIL (cargo test exit=$rc) ==="
  EXIT_CODE=2
fi

exit "$EXIT_CODE"
